#!/usr/bin/env python3
"""Insight PPE mission service (Route-A baseline).

- /api/ppe        : person + helmet open-vocab association
- /api/mission    : task-oriented PPE mission + evidence + event
- /api/events     : local event ledger
"""
from __future__ import annotations
import base64, io, json, os, re, sqlite3, threading, time, urllib.request
from urllib.parse import urlsplit, parse_qs
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL = "/cix/rexseek-studio/models/grounding-dino-tiny"
PROMPT = "person. safety helmet. hard hat. head."
BOX_TH = 0.22
TEXT_TH = 0.22
HELMET_MIN = 0.32
DB = "/cix/rexseek-studio/data/events.db"
QWEN = os.environ.get("QWEN_URL", "http://172.17.0.1:8081")
EVID = "/cix/rexseek-studio/web/data/evidence"
SEG = "/cix/rexseek-studio/web/data/segments"
ZONE_PATH = "/cix/rexseek-studio/data/zone.json"

# 一次入侵 = 一条事件：连续进入算同一事件；离开超过 GRACE_S 秒才确认结束。
GRACE_S = float(os.environ.get("GUARD_GRACE_S", "15"))
# 事件片段：进入期间逐帧归档，用于回放 + 整段分析（去重人数 / 安全帽）
SEG_MAX_FRAMES = int(os.environ.get("GUARD_SEG_FRAMES", "300"))
SEG_MAX_WIDTH = 960
DEFAULT_ZONE = [[0.28, 0.45], [0.78, 0.45], [0.88, 0.95], [0.18, 0.95]]
EPISODE_KEYS = ["id", "mission", "enter_ts", "exit_ts", "duration_s", "persons_max", "frames",
                "status", "evidence", "evidence_last", "sample", "video_enter", "video_exit",
                "session_id", "answer", "persons_total", "helmet_ok", "helmet_bad", "segment",
                "n_frames"]


def dur_text(seconds) -> str:
    s = int(max(0, round(float(seconds or 0))))
    if s < 60:
        return f"{s} 秒"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} 分 {sec} 秒" if sec else f"{m} 分钟"
    h, m2 = divmod(m, 60)
    return f"{h} 小时 {m2} 分"


def hms(iso: str | None) -> str | None:
    if not iso:
        return None
    return iso[11:19] if len(iso) >= 19 else iso


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def nms(boxes, scores, thr=0.55):
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return keep


class Service:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to("cuda").eval()
        self.dummy = Image.new("RGB", (336, 336), (128, 128, 128))
        self.run(self.dummy)
        os.makedirs(EVID, exist_ok=True)
        self.db = sqlite3.connect(DB, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mission TEXT, query TEXT,
            persons INTEGER, worn INTEGER, not_worn INTEGER, answer TEXT,
            evidence TEXT, latency_ms INTEGER)""")
        # 事件集：一条 = 一次完整的进入→离开（不再是每帧一条）
        self.db.execute("""CREATE TABLE IF NOT EXISTS guard_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission TEXT, enter_ts TEXT, exit_ts TEXT, duration_s INTEGER,
            persons_max INTEGER, frames INTEGER, status TEXT,
            evidence TEXT, evidence_last TEXT,
            sample TEXT, video_enter REAL, video_exit REAL,
            session_id TEXT, answer TEXT)""")
        self._ensure_columns()
        os.makedirs(SEG, exist_ok=True)
        try:                      # 让宿主机上的 server.py 也能写片段目录（ffmpeg 切片）
            os.chmod(os.path.dirname(SEG), 0o777)
            os.chmod(SEG, 0o777)
        except OSError:
            pass
        self.db.commit()
        self.guard = {}
        self.guard_lock = threading.Lock()
        self.agent_sessions = {}
        self._rebuild_episodes()
        self._reconcile_open()

    def _reconcile_open(self):
        """服务重启后，把上次没来得及收尾的“进行中”事件按最后一张取证图的时间关掉。"""
        try:
            rows = self.db.execute(
                "SELECT id, enter_ts, evidence_last FROM guard_episodes WHERE status!='closed'").fetchall()
            for eid, enter_ts, last_ev in rows:
                exit_ts = enter_ts
                m = re.search(r"tower_(\d{8})_(\d{6})_", last_ev or "")
                if m:
                    try:
                        exit_ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat(timespec="seconds")
                    except ValueError:
                        pass
                try:
                    dur = max(0, int((datetime.fromisoformat(exit_ts) -
                                      datetime.fromisoformat(enter_ts)).total_seconds()))
                except Exception:
                    dur = 0
                peak = self.db.execute("SELECT persons_max FROM guard_episodes WHERE id=?", (eid,)).fetchone()[0] or 0
                mission = self.db.execute("SELECT mission FROM guard_episodes WHERE id=?",
                                          (eid,)).fetchone()[0]
                kind = "duty" if mission == "duty_zone" else "danger"
                answer = self._episode_answer(kind, hms(enter_ts), hms(exit_ts), dur_text(dur), peak)
                self.db.execute(
                    "UPDATE guard_episodes SET status='closed', exit_ts=?, duration_s=?, answer=? WHERE id=?",
                    (exit_ts, dur, answer, eid))
                try:                     # 同时把片段统计（去重人数/安全帽）补上
                    self._finalize_segment(eid)
                except Exception:
                    pass
            if rows:
                self.db.commit()
                print(f"[guard] 收尾上次未结束的事件：{len(rows)} 条", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[guard] 收尾未结束事件失败：{exc}", flush=True)

    # ---------------------------------------------------------- 表结构 / 禁区
    def _ensure_columns(self):
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(guard_episodes)")}
        for name, typ in (("persons_total", "INTEGER"), ("helmet_ok", "INTEGER"),
                          ("helmet_bad", "INTEGER"), ("segment", "TEXT"), ("n_frames", "INTEGER")):
            if name not in cols:
                self.db.execute(f"ALTER TABLE guard_episodes ADD COLUMN {name} {typ}")
        self.db.commit()

    @staticmethod
    def _valid_zone(zone):
        if not isinstance(zone, (list, tuple)):
            return None
        pts = []
        try:
            for pt in zone:
                pts.append([round(float(pt[0]), 5), round(float(pt[1]), 5)])
        except (TypeError, ValueError, IndexError):
            return None
        if not (3 <= len(pts) <= 12):
            return None
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None
        return pts

    def get_zones(self):
        """危险区（有人=告警）与值守区（没人=告警）。"""
        try:
            with open(ZONE_PATH, encoding="utf-8") as fh:
                data = json.load(fh) or {}
            danger = self._valid_zone(data.get("danger") or data.get("zone")) or DEFAULT_ZONE
            duty = self._valid_zone(data.get("duty"))
            return {"danger": danger, "duty": duty}
        except (OSError, ValueError):
            return {"danger": DEFAULT_ZONE, "duty": None}

    def set_zones(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("需要 {danger: [...], duty: [...]}")
        danger = self._valid_zone(payload.get("danger") or payload.get("zone"))
        duty = self._valid_zone(payload.get("duty"))
        if payload.get("danger") is not None and not danger:
            raise ValueError("危险区需要 3~12 个 0~1 之间的点")
        if payload.get("duty") is not None and not duty:
            raise ValueError("值守区需要 3~12 个 0~1 之间的点")
        zone = {"danger": danger or DEFAULT_ZONE, "duty": duty,
                "updated": datetime.now().isoformat(timespec="seconds")}
        os.makedirs(os.path.dirname(ZONE_PATH), exist_ok=True)
        tmp = ZONE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(zone, fh, ensure_ascii=False)
        os.replace(tmp, ZONE_PATH)
        return {"danger": zone["danger"], "duty": zone["duty"]}

    def _resolve_zones(self, payload):
        if isinstance(payload.get("zones"), dict):
            z = payload["zones"]
            return {"danger": self._valid_zone(z.get("danger")), "duty": self._valid_zone(z.get("duty"))}
        legacy = self._valid_zone(payload.get("zone"))
        if legacy:
            return {"danger": legacy, "duty": None}
        return self.get_zones()

    # ------------------------------------------------- 安全帽关联（每帧）
    @staticmethod
    def _assoc_helmets(persons, helmets):
        used = set()
        out = []
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            head = [x1, y1, x2, y1 + (y2 - y1) * 0.38]
            best = None
            for i, hm in enumerate(helmets):
                if i in used:
                    continue
                hb = hm["box"]
                hx, hy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                if not (head[0] - 4 <= hx <= head[2] + 4 and head[1] - 4 <= hy <= head[3] + 4):
                    continue
                sane = ((hb[2] - hb[0]) <= (head[2] - head[0]) * 0.9
                        and (hb[3] - hb[1]) <= (head[3] - head[1]) * 1.2)
                if sane and (best is None or hm["score"] > best[0]):
                    best = (hm["score"], i)
            if best:
                used.add(best[1])
                out.append({**p, "helmet": True, "helmet_score": best[0]})
            else:
                out.append({**p, "helmet": False})
        return out

    # ------------------------------------------------------ 事件片段归档
    def _seg_dir(self, ep_id):
        return os.path.join(SEG, str(ep_id))

    def _save_seg_frame(self, ep_id, image, persons, wall_ts, video_t, seq):
        """把“进入期间”的每一帧存下来：回放证据 + 整段分析（去重人数/安全帽）"""
        try:
            d = self._seg_dir(ep_id)
            os.makedirs(d, exist_ok=True)
            try:
                os.chmod(d, 0o777)
            except OSError:
                pass
            w, h = image.size
            scale = min(1.0, SEG_MAX_WIDTH / max(w, h))
            thumb = image
            if scale < 1:
                thumb = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            thumb.save(os.path.join(d, f"f{seq:04d}.jpg"), quality=72)
            rec = {"seq": seq, "ts": round(wall_ts, 3), "video_t": video_t,
                   "size": [int(w), int(h)], "inside": len(persons),
                   "persons": [{"box": p["box"], "score": p["score"]} for p in persons]}
            with open(os.path.join(d, "frames.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[seg] 保存片段帧失败：{exc}", flush=True)

    @staticmethod
    def _track(frames, max_gap=4):
        """跨帧把同一个人串起来。

        抽帧间隔可能是 1~2 秒、人走动位移很大，单靠 IoU 会把同一个人拆成多条。
        这里用"上一帧位置 + 速度外推"预测位置，再按预测点距离匹配（IoU 作为补充）。
        """
        tracks = []
        for fr in frames:
            dets = fr.get("persons") or []
            size = fr.get("size") or [1280, 720]
            fw = float(size[0]) or 1280.0
            live = [t for t in tracks if fr["seq"] - t["last_seq"] <= max_gap]
            cand = []
            for di, det in enumerate(dets):
                box = det["box"]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                for ti, tr in enumerate(live):
                    dt = max(0.2, float(fr["ts"]) - tr["last_ts"])
                    px = tr["cx"] + tr.get("vx", 0.0) * dt
                    py = tr["cy"] + tr.get("vy", 0.0) * dt
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / fw
                    overlap = iou(box, tr["box"])
                    if overlap < 0.08 and dist > 0.35:
                        continue
                    cand.append((dist - overlap, di, ti, cx, cy, dt))
            cand.sort(key=lambda c: c[0])
            matched_d, matched_t = {}, set()
            for _cost, di, ti, cx, cy, dt in cand:
                if di in matched_d or ti in matched_t:
                    continue
                matched_d[di] = ti
                matched_t.add(ti)
                tr = live[ti]
                vx = (cx - tr["cx"]) / dt
                vy = (cy - tr["cy"]) / dt
                tr["vx"] = 0.5 * tr.get("vx", 0.0) + 0.5 * vx
                tr["vy"] = 0.5 * tr.get("vy", 0.0) + 0.5 * vy
                vt = fr.get("video_t")
                if vt is not None:
                    tr["vt0"] = min(tr["vt0"], vt) if tr.get("vt0") is not None else vt
                    tr["vt1"] = max(tr["vt1"], vt) if tr.get("vt1") is not None else vt
                tr.update({"box": dets[di]["box"], "cx": cx, "cy": cy, "last_seq": fr["seq"],
                           "last_ts": float(fr["ts"]), "hits": tr["hits"] + 1,
                           "helmet": tr["helmet"] + (1 if dets[di].get("helmet") else 0)})
            for di, det in enumerate(dets):
                if di in matched_d:
                    continue
                box = det["box"]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                tracks.append({"id": len(tracks) + 1, "first_ts": fr["ts"], "last_ts": fr["ts"],
                               "first_seq": fr["seq"], "last_seq": fr["seq"], "box": box,
                               "cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0,
                               "vt0": fr.get("video_t"), "vt1": fr.get("video_t"),
                               "hits": 1, "helmet": 1 if det.get("helmet") else 0})
        return tracks

    def _read_frames(self, ep_id):
        frames, path = [], os.path.join(self._seg_dir(ep_id), "frames.jsonl")
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frames.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return frames

    def _finalize_segment(self, ep_id):
        frames = self._read_frames(ep_id)
        if not frames:
            return None
        tracks = self._track(frames)
        min_hits = 2 if len(frames) >= 5 else 1
        valid = [t for t in tracks if t["hits"] >= min_hits]
        total = len(valid)
        summary = {
            "event_id": ep_id, "frames": len(frames), "persons_total": total,
            "helmet_ok": None, "helmet_bad": None, "helmet_disabled": True,
            "first_ts": frames[0]["ts"], "last_ts": frames[-1]["ts"],
            "video_enter": frames[0].get("video_t"), "video_exit": frames[-1].get("video_t"),
            "peak": max((f.get("inside") or 0) for f in frames),
            "tracks": [{"id": t["id"], "first_ts": t["first_ts"], "last_ts": t["last_ts"],
                        "hits": t["hits"]} for t in valid],
        }
        try:
            with open(os.path.join(self._seg_dir(ep_id), "segment.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=1)
        except OSError:
            pass
        self.db.execute(
            "UPDATE guard_episodes SET persons_total=?, helmet_ok=NULL, helmet_bad=NULL, segment=?, n_frames=? "
            "WHERE id=?", (total, f"data/segments/{ep_id}", len(frames), ep_id))
        row = self.db.execute(
            "SELECT enter_ts, exit_ts, duration_s, persons_max, mission FROM guard_episodes WHERE id=?",
            (ep_id,)).fetchone()
        if row:
            kind = "duty" if row[4] == "duty_zone" else "danger"
            if kind == "danger":
                base = self._episode_answer("danger", hms(row[0]), hms(row[1]), dur_text(row[2]), row[3])
                if total:
                    base += f"；期间共 {total} 人（跨帧去重）"
                self.db.execute("UPDATE guard_episodes SET answer=? WHERE id=?", (base, ep_id))
        self.db.commit()
        return summary

    def guard_segment(self, ep_id):
        ep = self._episode(int(ep_id))
        if not ep:
            return None
        frames = self._read_frames(int(ep_id))
        summary = None
        try:
            with open(os.path.join(self._seg_dir(ep_id), "segment.json"), encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            summary = None
        if not summary and frames:
            summary = self._finalize_segment(int(ep_id))
        items = []
        for fr in frames:
            try:
                ts = float(fr["ts"])
                hhmmss = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            except Exception:
                hhmmss = ""
            items.append({"seq": fr["seq"], "ts": fr["ts"], "hms": hhmmss,
                          "video_t": fr.get("video_t"), "inside": fr.get("inside"),
                          "url": f"/data/segments/{ep_id}/f{int(fr['seq']):04d}.jpg"})
        return {"event": ep, "summary": summary, "frames": items}

    def run(self, image: Image.Image, prompt: str = PROMPT):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        try:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=BOX_TH, text_threshold=TEXT_TH,
                target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=BOX_TH, text_threshold=TEXT_TH,
                target_sizes=[image.size[::-1]])[0]
        labels = res.get("text_labels")
        if labels is None:
            labels = res.get("labels") or ["object"] * len(res["boxes"])
        out = []
        for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            lab = str(lab).lower()
            if "helmet" in lab or "hard hat" in lab:
                if float(s) < HELMET_MIN:
                    continue
                kind = "helmet"
            elif "head" in lab:
                kind = "head"
            elif "person" in lab:
                kind = "person"
            else:
                continue
            out.append({"box": [round(float(v), 2) for v in b], "score": round(float(s), 4), "label": lab, "kind": kind})
        return out

    def ppe(self, data_url: str):
        t0 = time.perf_counter()
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        objs = self.run(image)
        persons = [o for o in objs if o["kind"] == "person"]
        helmets = [o for o in objs if o["kind"] == "helmet"]
        heads = [o for o in objs if o["kind"] == "head"]
        w, h = image.size
        if persons:
            pk = nms([p["box"] for p in persons], [p["score"] for p in persons], 0.42)
            persons = [persons[i] for i in pk]
        persons = [p for p in persons
                   if (p["box"][3] - p["box"][1]) >= h * 0.20
                   and (p["box"][2] - p["box"][0]) >= w * 0.04
                   and p["box"][1] >= 0.0]
        heads_roi = []
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            heads_roi.append([x1, y1, x2, y1 + (y2 - y1) * 0.38])
        pairs = []
        for pi, head in enumerate(heads_roi):
            for hi, hm in enumerate(helmets):
                hb = hm["box"]
                hx, hy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                inside = head[0] - 4 <= hx <= head[2] + 4 and head[1] - 4 <= hy <= head[3] + 4
                sane = (hb[2] - hb[0]) <= (head[2] - head[0]) * 0.9 and (hb[3] - hb[1]) <= (head[3] - head[1]) * 1.2
                if inside and sane:
                    pairs.append((hm["score"], pi, hi))
        pairs.sort(reverse=True)
        used_p, used_h = {}, set()
        for score, pi, hi in pairs:
            if pi in used_p or hi in used_h:
                continue
            used_p[pi] = hi
            used_h.add(hi)
        results = []
        for pi, p in enumerate(persons):
            hi = used_p.get(pi)
            hm = helmets[hi] if hi is not None else None
            results.append({
                "person": [round(v, 2) for v in p["box"]],
                "head": [round(v, 2) for v in heads_roi[pi]],
                "helmet": hm["box"] if hm else None,
                "status": "worn" if hm else "not_worn",
                "confidence": hm["score"] if hm else 0.5,
            })
        return {
            "size": [w, h], "persons": results, "helmets": helmets, "heads": heads,
            "summary": {"persons": len(persons), "helmets": len(helmets),
                        "worn": sum(1 for r in results if r["status"] == "worn"),
                        "not_worn": sum(1 for r in results if r["status"] == "not_worn")},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": "GroundingDINO-tiny open-vocab baseline",
        }

    def _draw(self, image: Image.Image, persons: list[dict]):
        canvas = image.copy()
        d = ImageDraw.Draw(canvas)
        for i, p in enumerate(persons, 1):
            box = p["person"]
            color = "#ff4d4f" if p["status"] == "not_worn" else "#2ee6a8"
            d.rectangle(box, outline=color, width=max(3, int(min(image.size) / 260)))
            label = f"{i} {'NO HELMET' if p['status'] == 'not_worn' else 'HELMET'}"
            d.rectangle([box[0], max(0, box[1] - 22), box[0] + 8 * len(label), box[1]], fill=color)
            d.text((box[0] + 4, max(0, box[1] - 20)), label, fill="#08111a")
            if p.get("helmet"):
                d.rectangle(p["helmet"], outline="#ffd84d", width=2)
        return canvas

    def mission_ppe(self, data_url: str, query: str):
        t0 = time.perf_counter()
        result = self.ppe(data_url)
        bad = [p for p in result["persons"] if p["status"] == "not_worn"]
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        canvas = self._draw(image, result["persons"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"ppe_{stamp}.jpg"
        path = os.path.join(EVID, name)
        canvas.save(path, quality=88)
        answer = (f"发现 {len(bad)} 名未佩戴安全帽人员" if bad else "未发现未佩戴安全帽人员")
        latency = int((time.perf_counter() - t0) * 1000)
        ev = f"/data/evidence/{name}"
        cur = self.db.execute(
            "INSERT INTO events(ts,mission,query,persons,worn,not_worn,answer,evidence,latency_ms) VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), "ppe_helmet", query,
             result["summary"]["persons"], result["summary"]["worn"], result["summary"]["not_worn"], answer, ev, latency))
        self.db.commit()
        return {
            "mission": "ppe_helmet", "query": query, "answer": answer,
            "summary": result["summary"], "findings": result["persons"],
            "rules": [{"name": "helmet_worn", "passed": not bad, "violations": len(bad)}],
            "evidence": ev, "event_id": cur.lastrowid,
            "metrics": {"perception_ms": result["latency_ms"], "rule_ms": 1, "total_ms": latency},
            "model": result["model"],
        }

    def events(self, limit=20):
        rows = self.db.execute("SELECT id,ts,mission,query,persons,worn,not_worn,answer,evidence,latency_ms FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ["id", "ts", "mission", "query", "persons", "worn", "not_worn", "answer", "evidence", "latency_ms"]
        return [dict(zip(keys, r)) for r in rows]

    def mission(self, payload):
        mission = payload.get("mission") or "ppe_helmet"
        query = payload.get("query") or "找出没戴安全帽的人"
        image = payload.get("image") or payload.get("data_url")
        if not image:
            raise ValueError("image data_url required")
        if mission in ("ppe_helmet", "helmet"):
            return self.mission_ppe(image, query)
        raise ValueError(f"unsupported mission: {mission}")

    @staticmethod
    def _inside(x, y, poly):
        if len(poly) < 3:
            return False
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _guard_draw(self, image, persons, zone, kind="danger"):
        canvas = image.copy()
        d = ImageDraw.Draw(canvas)
        w, h = image.size
        pts = [(int(x * w), int(y * h)) for x, y in (zone or [])]
        alarm = (kind == "danger" and len(persons) > 0) or (kind == "duty" and len(persons) == 0)
        zone_col = "#ff4d4f" if alarm else "#2ee6a8"
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=zone_col, width=max(3, int(min(w, h) / 170)))
        for i, p in enumerate(persons, 1):
            x1, y1, x2, y2 = p["box"]
            d.rectangle([x1, y1, x2, y2], outline=zone_col, width=max(3, int(min(w, h) / 220)))
            label = f"{i} PERSON"
            d.rectangle([x1, max(0, y1 - 22), x1 + 9 * len(label), y1], fill=zone_col)
            d.text((x1 + 4, max(0, y1 - 20)), label, fill="#08111a")
        tag = "DANGER: PERSON IN ZONE" if kind == "danger" else ("DUTY: NO ONE" if not persons else "DUTY: OK")
        d.rectangle([8, 8, 8 + 11 * len(tag), 30], fill="#08111a")
        d.text((12, 12), tag, fill=zone_col)
        return canvas

    def guard_frame(self, payload):
        t0 = time.perf_counter()
        image_url = payload.get("image") or payload.get("data_url")
        if not image_url:
            raise ValueError("image data_url required")
        zones = self._resolve_zones(payload)
        danger_zone, duty_zone = zones.get("danger"), zones.get("duty")
        sid = str(payload.get("session_id") or "default")
        now = float(payload.get("ts") or time.time())
        raw = image_url.split(",", 1)[1] if "," in image_url else image_url
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        # 只检测"人员"：安全帽等属性由用户在对话里提问时按需检测，值守过程不做自动属性标注
        objs = self.run(image, "person. worker.")
        persons = [o for o in objs if o["kind"] == "person"]
        w, h = image.size
        if persons:
            pk = nms([p["box"] for p in persons], [p["score"] for p in persons], 0.42)
            persons = [persons[i] for i in pk]
        persons = [p for p in persons if (p["box"][3] - p["box"][1]) >= h * 0.06]
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            cx, cy = (x1 + x2) / 2 / w, y2 / h
            p["in_danger"] = bool(danger_zone) and self._inside(cx, cy, danger_zone)
            p["in_duty"] = bool(duty_zone) and self._inside(cx, cy, duty_zone)
        danger_persons = [p for p in persons if p.get("in_danger")]
        duty_persons = [p for p in persons if p.get("in_duty")]

        sample = payload.get("sample") or None
        try:
            video_t = float(payload["video_t"]) if payload.get("video_t") is not None else None
        except (TypeError, ValueError):
            video_t = None

        with self.guard_lock:
            danger_ep, danger_status = self._step_zone(
                sid, "danger", danger_persons, image, danger_zone, now, video_t, sample)
            duty_ep, duty_status = self._step_zone(
                sid, "duty", duty_persons, image, duty_zone, now, video_t, sample,
                total_persons=len(persons))

        def alert(st):
            return st in ("alarm", "leaving")

        top_status = ("alarm" if (alert(danger_status) or alert(duty_status))
                      else ("watch" if not danger_ep and not duty_ep else "cleared"))
        evidence = (danger_ep or {}).get("evidence") or (duty_ep or {}).get("evidence")
        return {
            "status": top_status,
            "zones": zones,
            "danger": {"inside": len(danger_persons), "alert": alert(danger_status), "status": danger_status},
            "duty": {"inside": len(duty_persons), "empty": not duty_persons,
                     "alert": alert(duty_status), "status": duty_status},
            "persons": [{"box": p["box"], "score": p["score"],
                         "in_danger": bool(p.get("in_danger")), "in_duty": bool(p.get("in_duty")),
                         "inside": bool(p.get("in_danger"))} for p in persons],
            "inside": len(danger_persons),
            "event": danger_ep,
            "episode": danger_ep,
            "duty_event": duty_ep,
            "zones_alert": {"danger": alert(danger_status), "duty": alert(duty_status)},
            "evidence": evidence,
            "metrics": {"perception_ms": round((time.perf_counter() - t0) * 1000, 1)},
            "model": "GroundingDINO-tiny person + zone rule",
        }

    def _step_zone(self, sid, kind, persons_inside, image, zone, now, video_t, sample, total_persons=0):
        """危险区：区内有人=告警；值守区：区内没人=告警。同一种区域一次告警只记一条事件。"""
        if not zone:
            return None, "off"
        alarm_on_present = (kind == "danger")
        alarmed = (len(persons_inside) > 0) if alarm_on_present else (total_persons == 0 or len(persons_inside) == 0)
        if not alarm_on_present:
            alarmed = len(persons_inside) == 0
        key = f"{sid}::{kind}"
        st = self.guard.setdefault(key, {"ep": None, "enter": 0.0, "last_alarm": 0.0, "frames": 0,
                                         "peak": 0, "video_last": None, "seg_saved": 0})
        status = "watch"
        episode = None
        if alarmed:
            st["last_alarm"] = now
            st["last_alarm_dt"] = datetime.now()
            if alarm_on_present:
                st["peak"] = max(st.get("peak") or 0, len(persons_inside))
            if video_t is not None:
                st["video_last"] = video_t
            if st["ep"] is None:
                ev = self._save_evidence(image, persons_inside, zone, kind)
                enter_dt = st["last_alarm_dt"]
                count = len(persons_inside)
                st.update({"enter": now, "enter_dt": enter_dt, "frames": 1, "peak": count if alarm_on_present else 0})
                answer = self._open_answer(kind, enter_dt.isoformat(timespec="seconds"), count)
                mission = "danger_zone" if kind == "danger" else "duty_zone"
                cur = self.db.execute(
                    "INSERT INTO guard_episodes(mission,enter_ts,status,persons_max,frames,evidence,evidence_last,"
                    "sample,video_enter,video_exit,session_id,answer) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (mission, enter_dt.isoformat(timespec="seconds"), "open",
                     count if alarm_on_present else 0, 1, ev, ev, sample, video_t, video_t, sid, answer))
                self.db.commit()
                st["ep"] = cur.lastrowid
                st["seg_saved"] = 1
                self._save_seg_frame(st["ep"], image, persons_inside, now, video_t, 1)
            else:
                st["frames"] += 1
                last_ev = None
                if st["frames"] % 3 == 0:
                    last_ev = self._save_evidence(image, persons_inside, zone, kind)
                self.db.execute(
                    "UPDATE guard_episodes SET persons_max=?, frames=?, n_frames=?, "
                    "evidence_last=COALESCE(?,evidence_last), video_exit=COALESCE(?,video_exit), answer=? "
                    "WHERE id=?",
                    (st["peak"] if alarm_on_present else 0, st["frames"], min(st["frames"], SEG_MAX_FRAMES),
                     last_ev, video_t,
                     self._open_answer(kind, st["enter_dt"].isoformat(timespec="seconds"), st["peak"]), st["ep"]))
                self.db.commit()
                if st["frames"] <= SEG_MAX_FRAMES:
                    self._save_seg_frame(st["ep"], image, persons_inside, now, video_t, st["frames"])
                    st["seg_saved"] = st["frames"]
            status = "alarm"
            episode = self._episode(st["ep"])
        elif st["ep"]:
            if now - st["last_alarm"] > GRACE_S:
                episode = self._close_episode(st["ep"], st.get("last_alarm_dt") or datetime.now(),
                                              st.get("video_last"), st["peak"], kind)
                st.update({"ep": None, "enter": 0.0, "enter_dt": None, "frames": 0, "peak": 0,
                           "video_last": None, "last_alarm_dt": None, "seg_saved": 0})
                status = "cleared"
            else:
                status = "leaving"
                episode = self._episode(st["ep"])
        return episode, status

    # ------------------------------------------------------- 事件集（一次告警=一条）
    @staticmethod
    def _kind_label(kind):
        return "危险区" if kind == "danger" else "值守区"

    @staticmethod
    def _episode_answer(kind, enter_hms, exit_hms, dur_text_, peak):
        if kind == "danger":
            return (f"危险区入侵：{enter_hms} 进入，{exit_hms} 离开，持续 {dur_text_}，"
                    f"最多 {peak or 0} 人")
        return f"值守区无人：{enter_hms} 起无人值守，{exit_hms} 恢复，持续 {dur_text_}"

    def _save_evidence(self, image, persons, zone, kind="danger") -> str:
        canvas = self._guard_draw(image, persons, zone, kind)
        name = f"tower_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        canvas.save(os.path.join(EVID, name), quality=88)
        return f"/data/evidence/{name}"

    @staticmethod
    def _open_answer(kind, enter_ts: str, count: int) -> str:
        if kind == "danger":
            return f"危险区检测到 {count} 人进入（{hms(enter_ts)} 开始）"
        return f"值守区无人值守（{hms(enter_ts)} 开始）"

    def _close_episode(self, ep_id, last_alarm_dt, video_last, peak, kind="danger"):
        exit_ts = last_alarm_dt.isoformat(timespec="seconds")
        row = self.db.execute("SELECT enter_ts, mission FROM guard_episodes WHERE id=?", (ep_id,)).fetchone()
        enter_ts = (row[0] if row and row[0] else exit_ts)
        if row and row[1] == "duty_zone":
            kind = "duty"
        try:
            dur = max(0, int((datetime.fromisoformat(exit_ts) - datetime.fromisoformat(enter_ts)).total_seconds()))
        except Exception:
            dur = 0
        answer = self._episode_answer(kind, hms(enter_ts), hms(exit_ts), dur_text(dur), peak)
        self.db.execute(
            "UPDATE guard_episodes SET exit_ts=?, duration_s=?, status='closed', "
            "persons_max=COALESCE(?,persons_max), video_exit=COALESCE(?,video_exit), answer=? WHERE id=?",
            (exit_ts, dur, peak, video_last, answer, ep_id))
        self.db.commit()
        try:
            self._finalize_segment(ep_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[seg] 片段分析失败：{exc}", flush=True)
        return self._episode(ep_id)

    def _episode(self, ep_id):
        row = self.db.execute(
            "SELECT id,mission,enter_ts,exit_ts,duration_s,persons_max,frames,status,evidence,evidence_last,"
            "sample,video_enter,video_exit,session_id,answer,persons_total,helmet_ok,helmet_bad,segment,n_frames "
            "FROM guard_episodes WHERE id=?", (ep_id,)).fetchone()
        return self._episode_dict(row) if row else None

    @staticmethod
    def _episode_dict(row):
        d = dict(zip(EPISODE_KEYS, row))
        d["enter_hms"] = hms(d.get("enter_ts"))
        d["exit_hms"] = hms(d.get("exit_ts"))
        d["open"] = d.get("status") != "closed"
        if d["open"] and d.get("enter_ts"):
            # 进行中的事件：持续时长按“到现在”实时计算
            try:
                d["duration_s"] = max(0, int((datetime.now() - datetime.fromisoformat(d["enter_ts"])).total_seconds()))
            except Exception:
                d["duration_s"] = d.get("duration_s") or 0
        elif d.get("exit_ts") and not d.get("duration_s"):
            try:
                d["duration_s"] = max(0, int((datetime.fromisoformat(d["exit_ts"]) -
                                              datetime.fromisoformat(d["enter_ts"])).total_seconds()))
            except Exception:
                d["duration_s"] = 0
        d["duration_text"] = dur_text(d.get("duration_s") or 0)
        d["kind"] = "duty" if d.get("mission") == "duty_zone" else "danger"
        d["kind_label"] = "值守区" if d["kind"] == "duty" else "危险区"
        d["ts"] = d.get("enter_ts")          # 兼容旧字段名
        d["persons"] = d.get("persons_max") or 0
        return d

    def _rebuild_episodes(self):
        """把历史“每帧一条”的 events 台账合并成事件集（按 GRACE_S 间隔切分）。"""
        try:
            n = self.db.execute("SELECT COUNT(*) FROM guard_episodes").fetchone()[0]
            if n:
                return
            rows = self.db.execute(
                "SELECT id,ts,persons,evidence FROM events WHERE mission='tower_crane_zone' ORDER BY ts").fetchall()
            if not rows:
                return
            episodes, cur = [], None
            for _eid, ts, persons, ev in rows:
                try:
                    t = datetime.fromisoformat(ts)
                except Exception:
                    continue
                if cur and (t - cur["last"]).total_seconds() <= GRACE_S:
                    cur["last"] = t
                    cur["frames"] += 1
                    cur["peak"] = max(cur["peak"], persons or 0)
                else:
                    if cur:
                        episodes.append(cur)
                    cur = {"enter": t, "last": t, "frames": 1, "peak": persons or 0, "evidence": ev}
            if cur:
                episodes.append(cur)
            for e in episodes:
                dur = max(0, int((e["last"] - e["enter"]).total_seconds()))
                answer = (f"危险区入侵：{e['enter'].strftime('%H:%M:%S')} 进入，"
                          f"{e['last'].strftime('%H:%M:%S')} 离开，持续 {dur_text(dur)}，最多 {e['peak']} 人")
                self.db.execute(
                    "INSERT INTO guard_episodes(mission,enter_ts,exit_ts,duration_s,persons_max,frames,status,"
                    "evidence,evidence_last,answer) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("tower_crane_zone", e["enter"].isoformat(timespec="seconds"),
                     e["last"].isoformat(timespec="seconds"), dur, e["peak"], e["frames"], "closed",
                     e["evidence"], e["evidence"], answer))
            self.db.commit()
            print(f"[guard] 合并历史台账：{len(rows)} 帧 → {len(episodes)} 起事件", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[guard] 合并历史台账失败：{exc}", flush=True)

    def _qwen(self, messages, max_tokens=80):
        try:
            body = {"model": "qwen35-0.8b-cpu", "messages": messages, "max_tokens": max_tokens,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(QWEN.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read().decode())
            return (d["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            return ""

    def _plan_guard(self, query: str):
        """Qwen 做任务理解 + 工具规划；规则兜底。"""
        system = (
            "你是端侧区域值守 Agent 的规划器。根据用户问题只输出 JSON，不要解释：\n"
            '{"intent":"inspect|ledger|explain|followup|policy|chat","tools":["detect_scene","zone_check","temporal_check"]}\n'
            "inspect=现在画面/禁区是否有人；ledger=今天几次/次数；explain=为什么报警；"
            "followup=刚才那个/证据/放大；policy=配置规则；chat=闲聊。"
        )
        examples = [
            ("现在危险区里有人吗？", '{"intent":"inspect","tools":["detect_scene","zone_check"]}'),
            ("今天有几次违规？", '{"intent":"ledger","tools":["recall_events"]}'),
            ("为什么刚才报警？", '{"intent":"explain","tools":["explain_event"]}'),
            ("刚才那个放大看看", '{"intent":"followup","tools":["evidence"]}'),
            ("只要有人进入禁区就告警", '{"intent":"policy","tools":["parse_policy"]}'),
        ]
        msgs = [{"role": "system", "content": system}]
        for u, a in examples:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": query})
        raw = self._qwen(msgs, 80)
        try:
            import re
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            if data.get("intent") in ("inspect", "ledger", "explain", "followup", "policy", "chat"):
                return data, "qwen"
        except Exception:
            pass
        q = query
        if any(k in q for k in ("几次", "多少", "几起")):
            return {"intent": "ledger", "tools": ["recall_events"]}, "rule"
        if "为什么" in q or "原因" in q:
            return {"intent": "explain", "tools": ["explain_event"]}, "rule"
        if any(k in q for k in ("刚才", "证据", "放大", "看看")):
            return {"intent": "followup", "tools": ["evidence"]}, "rule"
        if any(k in q for k in ("有人", "禁区", "危险区", "值守区", "巡检")):
            return {"intent": "inspect", "tools": ["detect_scene", "zone_check"]}, "rule"
        return {"intent": "chat", "tools": []}, "rule"

    def guard_agent(self, payload):
        t0 = time.perf_counter()
        q = (payload.get("question") or payload.get("query") or "").strip()
        sid = str(payload.get("session_id") or "agent_default")
        plan, by = self._plan_guard(q)
        intent = plan["intent"]
        trace = [{"type": "thought", "text": f"识别任务：{intent}；按端侧工具链执行。"},
                 {"type": "plan", "steps": [{"tool": t, "chosen_by": by} for t in plan.get("tools", [])] or [{"tool": "chat", "chosen_by": by}]}]
        answer = ""
        objects = []
        evidence = None
        try:
            if intent == "inspect":
                image = payload.get("image") or payload.get("data_url")
                if not image:
                    raise ValueError("inspect 需要当前画面")
                t1 = time.perf_counter()
                obs = self.guard_frame({"image": image, "zone": payload.get("zone"), "session_id": sid,
                                        "ts": payload.get("ts")})
                trace.append({"type": "tool", "tool": "detect_scene", "status": "ok",
                              "observation": f"persons={len(obs.get('persons', []))} inside={obs.get('inside')} status={obs.get('status')}",
                              "duration_ms": int((time.perf_counter() - t1) * 1000), "objects": obs.get("persons", [])})
                objects = obs.get("persons", [])
                if obs.get("inside", 0):
                    answer = f"禁区内有 {obs['inside']} 人，当前状态：{obs['status']}。"
                    if obs.get("event"):
                        evidence = obs["event"].get("evidence")
                        answer += " 已抓拍并告警。"
                    else:
                        evidence = obs.get("evidence")
                else:
                    answer = "当前危险区内没有人员。"
                if not objects:
                    trace.append({"type": "recovery", "status": "warn", "text": "没有检测到可靠人员候选，不伪造告警。"})
            elif intent == "ledger":
                events = self.guard_events(30)
                total = self.guard_events_total()
                trace.append({"type": "tool", "tool": "recall_events", "status": "ok",
                              "observation": f"events={total}", "duration_ms": 1})
                if events:
                    latest = events[0]
                    answer = f"今天共 {total} 起区域告警事件。最近一起{self._episode_span(latest)}。"
                    evidence = latest.get("evidence")
                else:
                    answer = "今天还没有区域告警事件。"
            elif intent == "explain":
                events = self.guard_events(1)
                trace.append({"type": "tool", "tool": "explain_event", "status": "ok",
                              "observation": f"events={len(events)}", "duration_ms": 1})
                if events:
                    latest = events[0]
                    answer = (f"最近一起事件{self._episode_span(latest)}。"
                              f"规则：人员检测框底部中心落入区域多边形并持续即告警，"
                              f"离开超过 {int(GRACE_S)} 秒判定结束，因此一次进入只记一条。")
                    evidence = latest.get("evidence")
                else:
                    answer = "当前没有可解释的报警事件。"
            elif intent == "followup":
                events = self.guard_events(1)
                trace.append({"type": "tool", "tool": "evidence", "status": "ok",
                              "observation": f"events={len(events)}", "duration_ms": 1})
                if events:
                    latest = events[0]
                    answer = f"最近一起事件{self._episode_span(latest)}，已返回取证图。"
                    evidence = latest.get("evidence")
                else:
                    answer = "还没有可放大的历史事件。"
            elif intent == "policy":
                p = self.guard_policy({"text": q, "zone": payload.get("zone")})
                answer = f"已解析策略：{p.get('message')}（持续 {p.get('duration_s')}s）"
                trace.append({"type": "tool", "tool": "parse_policy", "status": "ok",
                              "observation": p.get("message", ""), "duration_ms": 1})
            else:
                answer = self._qwen([{"role": "system", "content": "你是端侧巡检助手，回答简洁，1-2句中文。"},
                                     {"role": "user", "content": q}], 80) or "我在。可以问我现在禁区内有没有人、今天几次违规，或者为什么报警。"
        except Exception as exc:
            trace.append({"type": "tool", "tool": "agent", "status": "error",
                          "observation": f"{type(exc).__name__}: {exc}", "duration_ms": 0, "objects": []})
            answer = "任务执行失败：没有可靠结果，请检查画面或稍后重试。"
        trace.append({"type": "final", "text": answer})
        self.agent_sessions[sid] = {"last_intent": intent, "last_answer": answer, "ts": time.time()}
        return {"answer": answer, "trace": trace, "objects": objects, "evidence": evidence,
                "intent": intent, "chosen_by": by,
                "metrics": {"total_ms": int((time.perf_counter() - t0) * 1000)}}

    def guard_events_query(self, limit=50, session_id=None):
        self._sweep_open()
        sql = ("SELECT id,mission,enter_ts,exit_ts,duration_s,persons_max,frames,status,evidence,evidence_last,"
               "sample,video_enter,video_exit,session_id,answer,persons_total,helmet_ok,helmet_bad,segment,n_frames "
               "FROM guard_episodes ")
        args = []
        if session_id:
            sql += "WHERE session_id=? "
            args.append(session_id)
        sql += "ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        rows = self.db.execute(sql, args).fetchall()
        return [self._episode_dict(r) for r in rows]

    def guard_events_total_for(self, session_id=None):
        if session_id:
            return int(self.db.execute("SELECT COUNT(*) FROM guard_episodes WHERE session_id=?", (session_id,)).fetchone()[0])
        return self.guard_events_total()

    @staticmethod
    def _event_time_line(e):
        status = "进行中" if e.get("open") else "已结束"
        span = f"{e.get('enter_hms') or '-'} → {e.get('exit_hms') or '—'}"
        return (f"#{e.get('id')} {span} · {e.get('duration_text') or '—'} · "
                f"最多 {e.get('persons_max') or 0} 人 · {status}")

    def guard_ask_filtered(self, payload):
        q = (payload.get("question") or payload.get("query") or "").strip()
        sid = payload.get("session_id")
        session_only = any(k in q for k in ("本会话", "当前会话", "这次会话", "本次会话", "我的会话", "这个会话"))
        if session_only and not sid:
            return {"answer": "当前页面还没有会话 ID，无法按会话筛选。", "events": [], "total": 0}
        show_all = any(k in q for k in ("所有", "全部", "列出", "明细", "每一条", "完整", "全部时间", "所有时间"))
        want_time = show_all or any(k in q for k in ("几点", "时间", "多久", "进入", "离开", "持续"))
        limit = 50 if show_all else 10
        events = self.guard_events_query(limit, sid if session_only else None)
        total = self.guard_events_total_for(sid if session_only else None)
        scope = "当前会话" if session_only else "今天"
        if not events:
            return {"answer": f"{scope}还没有区域告警事件。", "events": [], "evidence": None,
                    "total": 0, "session_filter": bool(session_only)}
        latest = events[0]
        if show_all or (want_time and ("所有" in q or "全部" in q or "列出" in q)):
            lines = [self._event_time_line(e) for e in events]
            body = f"{scope}共 {total} 起事件：\n" + "\n".join(lines)
            if total > len(lines):
                body += f"\n……仅显示最近 {len(lines)} 起"
            return {"answer": body, "events": events, "evidence": latest.get("evidence"),
                    "total": total, "session_filter": bool(session_only)}
        if any(k in q for k in ("几次", "多少", "几起")) or ("今天" in q and "几" in q) or ("会话" in q and "几" in q):
            return {"answer": f"{scope}共记录 {total} 起区域告警事件。最近一起{self._episode_span(latest)}。",
                    "events": events, "evidence": latest.get("evidence"), "total": total,
                    "session_filter": bool(session_only)}
        if session_only and any(k in q for k in ("事件", "记录", "列表", "筛选", "显示", "哪些")):
            lines = [self._event_time_line(e) for e in events]
            return {"answer": f"当前会话共 {total} 起事件：\n" + "\n".join(lines),
                    "events": events, "evidence": latest.get("evidence"), "total": total,
                    "session_filter": True}
        if "为什么" in q or "原因" in q or "怎么" in q:
            facts = (f"最近一起事件{self._episode_span(latest)}。规则：人员检测框底部中心落入区域多边形，"
                     f"且连续出现超过 0.45 秒即触发；离开超过 {int(GRACE_S)} 秒判定结束。")
            text = self._qwen([
                {"role": "system", "content": "你是端侧巡检助手。只根据给定事实回答，不要编造，1-2 句中文。"},
                {"role": "user", "content": facts + "\n回答：为什么触发报警？"},
            ], 80)
            return {"answer": text or facts, "events": events, "evidence": latest.get("evidence"),
                    "total": total, "session_filter": bool(session_only)}
        if any(k in q for k in ("刚才", "证据", "放大", "看看", "图")):
            return {"answer": f"最近一起事件{self._episode_span(latest)}，已附上进入时段的取证图片。",
                    "events": events, "evidence": latest.get("evidence"), "total": total,
                    "session_filter": bool(session_only)}
        if want_time:
            lines = [self._event_time_line(e) for e in events]
            return {"answer": f"{scope}共 {total} 起，时间如下：\n" + "\n".join(lines),
                    "events": events, "evidence": latest.get("evidence"), "total": total,
                    "session_filter": bool(session_only)}
        facts = f"最近一起事件{self._episode_span(latest)}；{scope}共 {total} 起。"
        text = self._qwen([
            {"role": "system", "content": "你是端侧巡检助手。只根据给定事实回答，不要编造，1-2 句中文。"},
            {"role": "user", "content": facts + "\n用户问：" + q},
        ], 80)
        return {"answer": text or facts, "events": events, "evidence": latest.get("evidence"),
                "total": total, "session_filter": bool(session_only)}

    def guard_ask(self, payload):
        q = (payload.get("question") or payload.get("query") or "").strip()
        events = self.guard_events(30)
        total = self.guard_events_total()
        if not events:
            return {"answer": "当前还没有区域告警事件。", "events": [], "evidence": None, "total": 0}
        latest = events[0]
        span = self._episode_span(latest)
        if any(k in q for k in ("几次", "多少", "几起")) or ("今天" in q and "几" in q):
            answer = f"今天共记录 {total} 起区域告警事件。最近一起{span}。"
            return {"answer": answer, "events": events[:10], "evidence": latest.get("evidence"), "total": total}
        if "为什么" in q or "原因" in q or "怎么" in q:
            facts = (f"最近一起事件{span}。规则：人员检测框底部中心落入区域多边形，"
                     f"且连续出现超过 0.45 秒即触发；离开超过 {int(GRACE_S)} 秒判定结束。")
            text = self._qwen([
                {"role": "system", "content": "你是端侧巡检助手。只根据给定事实回答，不要编造，1-2 句中文。"},
                {"role": "user", "content": facts + "\n回答：为什么触发报警？"},
            ], 80)
            return {"answer": text or facts, "events": events[:10], "evidence": latest.get("evidence"), "total": total}
        if any(k in q for k in ("刚才", "证据", "放大", "看看", "图")):
            return {"answer": f"最近一起事件{span}，已附上进入时段的取证图片。",
                    "events": events[:10], "evidence": latest.get("evidence"), "total": total}
        if any(k in q for k in ("几点", "什么时候", "时间", "多久", "几个人", "几人", "人数")):
            return {"answer": f"最近一起事件{span}。", "events": events[:10],
                    "evidence": latest.get("evidence"), "total": total}
        facts = f"最近一起事件{span}；今天共 {total} 起。"
        text = self._qwen([
            {"role": "system", "content": "你是端侧巡检助手。只根据给定事实回答，不要编造，1-2 句中文。"},
            {"role": "user", "content": facts + "\n用户问：" + q},
        ], 80)
        return {"answer": text or facts, "events": events[:10], "evidence": latest.get("evidence"), "total": total}

    @staticmethod
    def _episode_span(e: dict) -> str:
        if not e:
            return ""
        enter = e.get("enter_hms") or "-"
        peak = e.get("persons_max") or 0
        total = e.get("persons_total")
        extra = ""
        if total:
            extra = f"，期间共 {total} 人（去重）"
            if e.get("helmet_ok") is not None:
                extra += f"、安全帽 {e.get('helmet_ok')}/{total} 人佩戴"
        if e.get("kind") == "duty":
            return (f"：{enter} 起值守区无人，{e.get('exit_hms') or '仍在告警'} 恢复，"
                    f"持续 {e.get('duration_text') or '-'}")
        if e.get("exit_hms"):
            return (f"：{enter} 进入、{e['exit_hms']} 离开，持续 {e.get('duration_text') or '-'}，"
                    f"最多 {peak} 人{extra}")
        return f"：{enter} 进入，值守中（已持续 {e.get('duration_text') or '-'}，最多 {peak} 人{extra}）"

    def guard_events(self, limit=30):
        self._sweep_open()
        rows = self.db.execute(
            "SELECT id,mission,enter_ts,exit_ts,duration_s,persons_max,frames,status,evidence,evidence_last,"
            "sample,video_enter,video_exit,session_id,answer,persons_total,helmet_ok,helmet_bad,segment,n_frames "
            "FROM guard_episodes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._episode_dict(r) for r in rows]

    def _sweep_open(self):
        """页面关掉/停止值守后，没人来收尾的事件按最后取证时间自动结束。

        判定只看"最后一次取证图的时间"：停止值守后不再有新帧，超过 GRACE_S+10 秒
        就按最后一张取证图的时刻收尾。人在画面里连续值守时，取证图每 3 帧刷新一次，
        不会误判。
        """
        try:
            rows = self.db.execute(
                "SELECT id, enter_ts, evidence_last, persons_max FROM guard_episodes WHERE status!='closed'").fetchall()
            if not rows:
                return
            now = datetime.now()
            swept = []
            for eid, enter_ts, last_ev, peak in rows:
                last = None
                m = re.search(r"tower_(\d{8})_(\d{6})_", last_ev or "")
                if m:
                    try:
                        last = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                    except ValueError:
                        last = None
                if last is None:
                    try:
                        last = datetime.fromisoformat(enter_ts)
                    except Exception:
                        last = now
                if (now - last).total_seconds() <= GRACE_S + 10:
                    continue
                self._close_episode(eid, last, None, peak or 0)
                swept.append(eid)
            if swept:
                with self.guard_lock:
                    for st in self.guard.values():
                        if st.get("ep") in swept:
                            st.update({"ep": None, "enter": 0.0, "enter_dt": None, "frames": 0,
                                       "peak": 0, "video_last": None, "last_inside_dt": None})
        except Exception:  # noqa: BLE001
            pass

    def guard_events_total(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM guard_episodes").fetchone()[0])

    def guard_event(self, ep_id) -> dict | None:
        return self._episode(int(ep_id))

    def guard_policy(self, payload):
        text = payload.get("text") or "只要有人进入危险区，就喊话、抓拍并告警"
        zone = payload.get("zone") or [[0.28, 0.45], [0.78, 0.45], [0.88, 0.95], [0.18, 0.95]]
        message = "危险区域禁止停留，请立即离开。"
        # 轻量 LLM：解释/生成告警话术；规则参数保持确定性
        try:
            body = {
                "model": "qwen35-0.8b-cpu",
                "messages": [
                    {"role": "system", "content": "你是端侧巡检策略助手。只输出一句不超过30字的中文告警话术，不要解释。"},
                    {"role": "user", "content": f"场景：工地危险区域。策略：{text}"},
                ],
                "max_tokens": 40, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(QWEN.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read().decode())
                message = (d["choices"][0]["message"].get("content") or message).strip()
        except Exception:
            pass
        return {"mission": "tower_crane_zone", "zone": zone, "duration_s": 0.45,
                "message": message, "condition": {"object": "person", "inside": "zone"},
                "actions": ["broadcast", "capture_evidence", "alert"]}


HTML = """<!doctype html><meta charset=utf-8><title>Insight Mission</title>
<style>body{background:#08111a;color:#d8f3ff;font:14px system-ui;padding:24px}pre{white-space:pre-wrap}</style>
<h2>Insight Mission API</h2><p>请使用 /insight.html 控制台。</p>"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    svc = None

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] " + fmt % args, flush=True)

    def _json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            return self._json({"ok": True, "service": "insight-ppe", "model": "GroundingDINO-tiny",
                               "missions": ["ppe_helmet"], "grace_s": GRACE_S})
        if self.path.startswith("/api/events"):
            return self._json({"events": self.svc.events()})
        if self.path.startswith("/api/guard/event/"):
            try:
                eid = int(self.path.rstrip("/").rsplit("/", 1)[-1])
            except ValueError:
                return self._json({"error": "bad id"}, 400)
            ep = self.svc.guard_event(eid)
            if not ep:
                return self._json({"error": "not found"}, 404)
            return self._json({"event": ep, "total": self.svc.guard_events_total()})
        if self.path.split("?", 1)[0] == "/api/guard/events":
            parts = urlsplit(self.path)
            params = parse_qs(parts.query)
            sid = (params.get("session_id") or [None])[0]
            try:
                limit = max(1, min(200, int((params.get("limit") or ["50"])[0])))
            except ValueError:
                limit = 50
            return self._json({"events": self.svc.guard_events_query(limit, sid),
                               "total": self.svc.guard_events_total_for(sid),
                               "session_filter": bool(sid), "session_id": sid})
        if self.path == "/api/guard/zone":
            zones = self.svc.get_zones()
            return self._json({"zone": zones.get("danger"), **zones})
        if self.path.startswith("/api/guard/segment/"):
            try:
                sid = int(self.path.rstrip("/").rsplit("/", 1)[-1])
            except ValueError:
                return self._json({"error": "bad id"}, 400)
            seg = self.svc.guard_segment(sid)
            if not seg:
                return self._json({"error": "not found"}, 404)
            return self._json(seg)
        if self.path == "/":
            raw = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            return
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/api/ppe", "/api/mission", "/api/guard/frame", "/api/guard/policy",
                             "/api/guard/ask", "/api/guard/agent", "/api/guard/zone"):
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode())
            if self.path == "/api/guard/zone":
                zones = self.svc.set_zones(payload)
                return self._json({"zone": zones.get("danger"), **zones})
            if self.path == "/api/guard/frame":
                return self._json(self.svc.guard_frame(payload))
            if self.path == "/api/guard/ask":
                return self._json(self.svc.guard_ask_filtered(payload))
            if self.path == "/api/guard/agent":
                return self._json(self.svc.guard_agent(payload))
            if self.path == "/api/guard/policy":
                return self._json(self.svc.guard_policy(payload))
            if self.path == "/api/ppe":
                img = payload.get("image") or payload.get("data_url")
                if not img:
                    raise ValueError("image data_url required")
                return self._json(self.svc.ppe(img))
            return self._json(self.svc.mission(payload))
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


if __name__ == "__main__":
    H.svc = Service()
    srv = ThreadingHTTPServer(("0.0.0.0", 8092), H)
    print("Insight PPE mission service on :8092", flush=True)
    srv.serve_forever()
