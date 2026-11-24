#!/usr/bin/env python3
"""RexSeek Studio 本地服务。

- 静态资源：web/
- GET  /api/health   引擎状态
- GET  /api/samples  示例素材元数据
- POST /api/chat     对话 + 定位

只依赖 Python 标准库，直接运行即可：

    python3 server.py                 # 演示模式（内置样例）
    python3 server.py --port 8848     # 指定端口
    python3 server.py --engine proxy --endpoint http://gpu-host:9000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
CATALOG_PATH = ROOT / "data" / "samples.json"
MEDIA_PATH = ROOT / "data" / "media.json"
MEDIA_DIR = WEB / "media"
MEDIA_MAX_KEEP = 5
MEDIA_MAX_BYTES = 120 * 1024 * 1024
INCLUDE_MEDIA = True

GUARD_SUGGESTIONS = [
    "现在危险区内有人吗？",
    "今天有几起告警事件？",
    "刚才那起事件是怎么回事？",
]

with CATALOG_PATH.open(encoding="utf-8") as fh:
    CATALOG = json.load(fh)

from engine import create_engine  # noqa: E402  (需要先读 catalog 给类型提示)

ENGINE = None  # 在 main() 中初始化
VERSION = "1.1.0"
STARTED = time.time()
AGENT_ENDPOINT = None
YOLOE_ENDPOINT = os.environ.get("YOLOE_URL", "http://127.0.0.1:8090")


def load_media() -> list:
    try:
        with MEDIA_PATH.open(encoding="utf-8") as fh:
            return json.load(fh).get("media", []) or []
    except (OSError, ValueError):
        return []


def save_media(records: list) -> None:
    MEDIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEDIA_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump({"media": records}, fh, ensure_ascii=False, indent=2)
    tmp.replace(MEDIA_PATH)


def media_sample(m: dict) -> dict:
    return {
        "id": m["id"],
        "kind": "video",
        "title": m.get("title") or "现场视频 · 塔吊禁区",
        "subtitle": m.get("subtitle") or "本地视频 · 逐帧值守",
        "file": m["file"],
        "poster": m.get("poster"),
        "width": m.get("width") or 1280,
        "height": m.get("height") or 720,
        "duration": m.get("duration") or 0,
        "suggestions": GUARD_SUGGESTIONS,
        "live_suggestions": GUARD_SUGGESTIONS,
        "objectCount": 0,
    }


def sample_map() -> dict:
    out = {}
    for m in load_media():
        out[m["id"]] = m
    for s in CATALOG.get("samples", []):
        out[s["id"]] = {"id": s["id"], "file": s["file"], "kind": s.get("kind"),
                        "width": s.get("width"), "height": s.get("height")}
    return out


def _probe(url: str, timeout: float = 1.2) -> dict:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read().decode() or "{}")
        return {"ok": True, "ms": int((time.perf_counter() - t0) * 1000), "info": body}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ms": int((time.perf_counter() - t0) * 1000), "error": str(exc)[:120]}


def status_payload() -> dict:
    targets = {
        "ppe": (PPE_ENDPOINT or "") .rstrip("/") + "/api/health" if PPE_ENDPOINT else "",
        "agent": (AGENT_ENDPOINT or "").rstrip("/") + "/api/health" if AGENT_ENDPOINT else "",
        "yoloe": (YOLOE_ENDPOINT or "").rstrip("/") + "/health",
    }
    results = {}
    threads = []
    for name, url in targets.items():
        if not url:
            results[name] = {"ok": False, "error": "未配置"}
            continue

        def work(n=name, u=url):
            results[n] = _probe(u)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout=2.0)
    services = {"gateway": {"ok": True, "ms": 0, "info": {"engine": ENGINE.name if ENGINE else "?"}}}
    services.update(results)
    return {
        "ok": all(v.get("ok") for v in services.values()),
        "version": VERSION,
        "uptime_s": int(time.time() - STARTED),
        "model": CATALOG.get("model", "RexSeek-3B"),
        "samples": len(public_samples()["samples"]),
        "services": services,
    }


def cut_clip(event: dict) -> str | None:
    """把事件对应的源视频时段切成 mp4 证据片段（H.264，浏览器可直接播）。"""
    sample_id = event.get("sample")
    v0 = event.get("video_enter")
    v1 = event.get("video_exit")
    if not sample_id or v0 is None or v1 is None:
        return None
    meta = sample_map().get(sample_id)
    if not meta:
        return None
    src = WEB / meta["file"]
    if not src.exists():
        return None
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg):
        return None
    # 片段目录由 ppe_service（容器内 root）创建，宿主机进程写不进去，单独放 clips/
    out_dir = WEB / "data" / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{event['id']}.mp4"
    start = max(0.0, float(v0) - 0.3)
    dur = max(0.6, float(v1) - float(v0) + 0.6)
    cmd = [ffmpeg, "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
           "-vf", "scale=960:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return None
    return f"/data/clips/{event['id']}.mp4"


def public_samples() -> dict:
    # 本地导入的视频排在最前面 → 两个页面的“默认例子”就是它
    samples = [media_sample(m) for m in load_media()] if INCLUDE_MEDIA else []
    for s in CATALOG["samples"]:
        samples.append(
            {
                "id": s["id"],
                "kind": s["kind"],
                "title": s["title"],
                "subtitle": s["subtitle"],
                "file": s["file"],
                "width": s["width"],
                "height": s["height"],
                "duration": s.get("duration"),
                "poster": s.get("poster"),
                "suggestions": s.get("suggestions", []),
                "live_suggestions": s.get("live_suggestions", []),
                "objectCount": len(s.get("objects", [])),
            }
        )
    return {"model": CATALOG.get("model", "RexSeek-3B"), "samples": samples}


class Handler(SimpleHTTPRequestHandler):
    server_version = "RexSeekStudio/1.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    # ---------------------------------------------------------------- utils
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def end_headers(self):
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/api/") or path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----------------------------------------------------------- 素材上传
    @staticmethod
    def _b64(raw: str) -> bytes:
        return base64.b64decode(raw + "=" * (-len(raw) % 4))

    def _upload_media(self):
        """把浏览器里的 mp4 固化成本地素材：两页默认例子都用它，事件跳转才能定位到视频时间。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._json({"error": "empty body"}, 400)
        if length > int(MEDIA_MAX_BYTES * 1.38) + 4 * 1024 * 1024:
            return self._json({"error": "文件太大（上限 120MB）"}, 413)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json({"error": "invalid json"}, 400)
        if not isinstance(payload, dict):
            return self._json({"error": "invalid payload"}, 400)

        data_url = str(payload.get("data_url") or "")
        if not data_url.startswith("data:") or "," not in data_url:
            return self._json({"error": "data_url required"}, 400)
        head, encoded = data_url.split(",", 1)
        try:
            blob = self._b64(encoded)
        except Exception:  # noqa: BLE001
            return self._json({"error": "bad base64"}, 400)
        if not blob:
            return self._json({"error": "empty file"}, 400)
        if len(blob) > MEDIA_MAX_BYTES:
            return self._json({"error": "文件太大（上限 120MB）"}, 413)

        name = str(payload.get("name") or "video.mp4")
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".mp4", ".webm", ".mov", ".m4v", ".ogv"):
            ext = ".mp4" if "video" in head else ".jpg"
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        ident = "site_" + time.strftime("%Y%m%d_%H%M%S")
        while (MEDIA_DIR / (ident + ext)).exists():
            ident += "x"
        (MEDIA_DIR / (ident + ext)).write_bytes(blob)

        poster_file = None
        poster = str(payload.get("poster") or "")
        if poster.startswith("data:") and "," in poster:
            try:
                pblob = self._b64(poster.split(",", 1)[1])
                if pblob:
                    pname = ident + ".jpg"
                    (MEDIA_DIR / pname).write_bytes(pblob)
                    poster_file = f"media/{pname}"
            except Exception:  # noqa: BLE001
                poster_file = None

        record = {
            "id": ident,
            "file": f"media/{ident}{ext}",
            "poster": poster_file,
            "title": str(payload.get("title") or "现场视频 · 塔吊禁区"),
            "subtitle": str(payload.get("subtitle") or "本地视频 · 逐帧值守"),
            "width": int(payload.get("width") or 1280),
            "height": int(payload.get("height") or 720),
            "duration": round(float(payload.get("duration") or 0), 2),
            "size": len(blob),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        records = [r for r in load_media() if r.get("id") != ident]
        records.insert(0, record)
        keep, drop = records[:MEDIA_MAX_KEEP], records[MEDIA_MAX_KEEP:]
        for old in drop:
            for rel in (old.get("file"), old.get("poster")):
                if not rel:
                    continue
                try:
                    (WEB / rel).unlink()
                except OSError:
                    pass
        save_media(keep)
        return self._json({"ok": True, "sample": media_sample(record), "count": len(keep)})

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/health":
            info = ENGINE.describe()
            info.update({"ok": True, "version": "1.0", "samples": len(CATALOG["samples"])})
            return self._json(info)
        if path == "/api/samples":
            return self._json(public_samples())
        if path == "/api/status":
            return self._json(status_payload())
        if path == "/api/guard/clip/":
            return self._json({"error": "not found"}, 404)
        if path.startswith("/api/guard/clip/"):
            if not PPE_ENDPOINT:
                return self._json({"error": "PPE endpoint not configured"}, 400)
            try:
                eid = path.rstrip("/").rsplit("/", 1)[-1]
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + f"/api/guard/event/{eid}",
                                            timeout=10) as resp:
                    event = json.loads(resp.read().decode()).get("event") or {}
                clip = cut_clip(event)
                if not clip:
                    return self._json({"error": "该事件没有可切片的源视频"}, 404)
                return self._json({"clip": clip, "event_id": event.get("id")})
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 502)
        if path == "/api/guard/zone":
            if not PPE_ENDPOINT:
                return self._json({"zone": None})
            try:
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + "/api/guard/zone", timeout=8) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                return self._json({"zone": None, "error": str(exc)}, 502)
        if path.startswith("/api/guard/segment/"):
            if not PPE_ENDPOINT:
                return self._json({"error": "PPE endpoint not configured"}, 400)
            try:
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + path, timeout=20) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 502)
        if path == "/api/media":
            return self._json({"media": load_media(), "samples": [media_sample(m) for m in load_media()]})
        if path.startswith("/api/guard/event/"):
            if not PPE_ENDPOINT:
                return self._json({"error": "PPE endpoint not configured"}, 400)
            try:
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + path, timeout=10) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 502)
        if path == "/api/ppe/health":
            return self._json({"ok": bool(PPE_ENDPOINT), "endpoint": PPE_ENDPOINT})
        if path == "/api/guard/events":
            if not PPE_ENDPOINT:
                return self._json({"events": []})
            try:
                qs = urllib.parse.urlsplit(self.path).query
                suffix = ("?" + qs) if qs else ""
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + "/api/guard/events" + suffix, timeout=10) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                return self._json({"events": [], "error": str(exc)}, 502)
        if path == "/api/events":
            if not PPE_ENDPOINT:
                return self._json({"events": []})
            try:
                with urllib.request.urlopen(PPE_ENDPOINT.rstrip("/") + "/api/events", timeout=10) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                return self._json({"events": [], "error": str(exc)}, 502)
        if path.startswith("/api/session/"):
            if not hasattr(ENGINE, "get_session"):
                return self._json({"session_id": path.rsplit("/", 1)[-1], "exists": False}, 404)
            try:
                return self._json(ENGINE.get_session(path.rsplit("/", 1)[-1]))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 502)
        return super().do_GET()

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/api/ppe", "/api/mission", "/api/guard/frame", "/api/guard/policy", "/api/guard/ask",
                    "/api/guard/agent", "/api/guard/zone"):
            if not PPE_ENDPOINT:
                return self._json({"error": "PPE endpoint not configured"}, 400)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                req = urllib.request.Request(PPE_ENDPOINT.rstrip("/") + path, data=raw,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": str(exc)}, 502)

        if path == "/api/ppe_old":
            if not PPE_ENDPOINT:
                return self._json({"error": "PPE endpoint not configured"}, 400)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                req = urllib.request.Request(PPE_ENDPOINT.rstrip("/") + "/api/ppe", data=raw,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return self._json(json.loads(resp.read().decode()))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": str(exc)}, 502)

        if path == "/api/media/upload":
            return self._upload_media()

        if path not in ("/api/chat", "/api/chat/stream", "/api/index", "/api/session"):
            return self._json({"error": "not found"}, 404)

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._json({"error": "empty body"}, 400)
        if length > 32 * 1024 * 1024:
            return self._json({"error": "payload too large"}, 413)

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json({"error": "invalid json"}, 400)
        if not isinstance(payload, dict):
            return self._json({"error": "invalid payload"}, 400)

        if path == "/api/session":
            if not hasattr(ENGINE, "create_session"):
                return self._json({"error": "session is only available in proxy mode"}, 400)
            try:
                return self._json(ENGINE.create_session(payload))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)}, 502)

        if path == "/api/index":
            if not hasattr(ENGINE, "index"):
                return self._json({"error": "index is only available in proxy mode"}, 400)
            try:
                return self._json(ENGINE.index(payload))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": str(exc)}, 502)

        if path == "/api/chat/stream":
            if not hasattr(ENGINE, "open_stream"):
                return self._json({"error": "stream is only available in proxy mode"}, 400)
            try:
                upstream = ENGINE.open_stream(payload)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": str(exc)}, 502)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    chunk = upstream.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                upstream.close()
            return

        started = time.perf_counter()
        try:
            result = ENGINE.respond(payload)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": str(exc)}, 502)

        meta = result.setdefault("meta", {})
        meta["latency_ms"] = int((time.perf_counter() - started) * 1000)
        meta.setdefault("engine", ENGINE.name)
        meta.setdefault("model", ENGINE.model)
        result.setdefault("objects", [])
        return self._json(result)

    # ------------------------------------------------------------- 静态文件
    def send_head(self):
        """带 Range 支持，保证视频可以拖动进度条。"""
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith("/"):
                self.send_response(301)
                self.send_header("Location", parts.path + "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            for index in ("index.html", "index.htm"):
                candidate = os.path.join(path, index)
                if os.path.exists(candidate):
                    path = candidate
                    break
            else:
                return self.list_directory(path)

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            size = os.fstat(f.fileno()).st_size
            range_header = self.headers.get("Range", "")
            self._range_length = None
            if range_header.startswith("bytes=") and size > 0:
                match = re.match(r"bytes=(\d*)-(\d*)$", range_header.split(",")[0].strip())
                if match:
                    start = int(match.group(1)) if match.group(1) else 0
                    end = int(match.group(2)) if match.group(2) else size - 1
                    end = min(end, size - 1)
                    if start >= size or start > end:
                        f.close()
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return None
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    f.seek(start)
                    self._range_length = length
                    return f

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_length", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description="RexSeek Studio local server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，局域网演示可用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--engine", default="mock", choices=["mock", "proxy"])
    parser.add_argument("--endpoint", default=None, help="proxy 引擎的推理服务地址")
    parser.add_argument("--timeout", type=float, default=180, help="proxy 请求超时（秒）")
    parser.add_argument("--ppe-endpoint", default=None, help="PPE baseline 服务地址")
    parser.add_argument("--samples", default=None, help="使用独立的示例 catalog（通用视觉演示）")
    parser.add_argument("--no-media", action="store_true", help="不把本地值守视频加入示例列表")
    args = parser.parse_args()

    global ENGINE, PPE_ENDPOINT, AGENT_ENDPOINT, CATALOG, CATALOG_PATH, INCLUDE_MEDIA
    PPE_ENDPOINT = args.ppe_endpoint
    AGENT_ENDPOINT = args.endpoint
    if args.samples:
        CATALOG_PATH = Path(args.samples)
        with CATALOG_PATH.open(encoding="utf-8") as fh:
            CATALOG = json.load(fh)
    INCLUDE_MEDIA = not args.no_media
    ENGINE = create_engine(args.engine, CATALOG, endpoint=args.endpoint, timeout=args.timeout)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print("")
    print("  RexSeek Studio")
    print(f"  → http://{shown_host}:{args.port}")
    print(f"  → engine: {ENGINE.name}")
    print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
