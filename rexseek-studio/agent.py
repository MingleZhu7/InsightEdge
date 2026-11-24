"""RexSeek Agent：任务规划、工具调度、会话记忆与异常恢复。

设计目标不是"再包一层 if/else"，而是把一次请求拆成可观察的
Thought -> Plan -> Tool -> Observation -> Final 轨迹：

- 规划：按“先/再/然后”等线索拆分子任务；每个子任务交给 Qwen 做工具分类，
  规则做兜底校验（1.5B 的分类结果不稳定，不能盲信）。
- 工具：ground / count / describe / relative / resolve / clarify。
- 记忆：按 session_id 保存最近目标；支持“她/他/第二个/刚才那个”。
- 恢复：工具异常重试；ground 为空时降级到统计/描述；相对位置失败时回退全图。
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime
import io
import json
import os
import re
import time
import urllib.request

from session_store import SESSION_RE
from typing import Any

PALETTE = ["#35e0ff", "#ffb547", "#a78bfa", "#b6f36b", "#ff7a6b", "#7cc4ff", "#ff8fd8", "#ffd166"]
LABEL_ZH = {"person": "人", "tomato": "番茄", "bus": "大巴", "car": "汽车", "motorcycle": "摩托车", "truck": "卡车"}
PPE_URL = os.environ.get("PPE_URL", "http://127.0.0.1:8092")
DEFAULT_ZONE = [[0.28, 0.45], [0.78, 0.45], [0.88, 0.95], [0.18, 0.95]]

GREET = ("你好", "您好", "hi", "hello", "嗨", "在吗", "早上好", "晚上好")
THANKS = ("谢谢", "感谢", "thanks", "thankyou")
CAP = ("你能做什么", "你会什么", "怎么用", "帮助", "help", "你是谁", "介绍一下你自己", "自我介绍")
DESCRIBE = ("描述", "介绍一下这个画面", "画面里有什么", "有什么东西", "整体", "总结一下")
COUNT = ("几个", "几颗", "多少", "数量", "数一数", "几个人", "几辆车")
DIRS = {"左": "left", "右": "right", "上": "above", "下": "below", "旁": "side"}


def norm(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?~…]+", "", (text or "").lower())


def match_intent(sample: dict | None, query: str) -> dict | None:
    if not sample:
        return None
    q = norm(query)
    best, best_score = None, 0
    for intent in sample.get("intents", []):
        score = sum(max(2, len(norm(k))) for k in intent.get("keywords", []) if norm(k) and norm(k) in q)
        if score > best_score:
            best, best_score = intent, score
    return best


def spatial(box) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = "左侧" if cx < 0.34 else ("右侧" if cx > 0.66 else "中部")
    v = "偏上" if cy < 0.34 else ("偏下" if cy > 0.66 else "中部")
    if h == "中部" and v == "中部":
        return "画面中央"
    if h == "中部":
        return f"画面{v}"
    if v == "中部":
        return f"画面{h}"
    return f"画面{h}{v}"


def strip_question(text: str) -> str:
    t = text.strip(" ，,；;。")
    t = re.sub(r"^(请|帮我|给我|先|再|然后|接着|最后|接下来|随后|麻烦|告诉我|找出|定位|看看|看一下|找一找)+", "", t)
    t = re.sub(r"^(画面里|图片里|图里|照片里|这|那|有|有没有|是不是)+", "", t)
    t = re.sub(r"(在哪里|在哪儿|在哪|是谁|是什么|呢|吗|？|\?)+$", "", t).strip()
    return t or text


class RexSeekAgent:
    def __init__(self, runtime, samples: dict, callbacks: dict, store=None):
        self.rt = runtime
        self.samples = samples
        self.cb = callbacks  # palette/spatial 之类的小工具
        self.store = store
        self.sessions: dict[str, dict] = {}
        self.answer_cache: dict[tuple, dict] = {}

    # ----------------------------------------------------------------- 入口
    def handle(self, payload: dict, emit=None) -> dict:
        started = time.perf_counter()
        query = self._last_user(payload)
        print(f"[agent] start q={query[:50]!r} session={payload.get('session_id','default')}", flush=True)
        media = payload.get("media") or {}
        sample = self.samples.get(media.get("sample_id") or "")
        sid = payload.get("session_id") or "default"
        if not SESSION_RE.match(sid):
            sid = "default"
        session = self._session(sid)
        session["_sid"] = sid
        # 从值守台台账点进来的事件：后续追问都针对这一起，而不是"最近一起"
        try:
            eid = payload.get("event_id")
            session["event_id"] = int(eid) if eid not in (None, "", "null") else None
        except (TypeError, ValueError):
            session["event_id"] = None
        image = self.rt._load_image(media, sample)
        image_key = getattr(image, "_rexseek_hash", None)
        if image_key:
            if session.get("image_key") and session["image_key"] != image_key:
                session["last_subject"] = None
                session["last_objects"] = []
                session["memory_reset"] = True
            session["image_key"] = image_key
        cache_key = self._answer_key(image_key, query, session.get("event_id"))
        if cache_key and cache_key in self.answer_cache:
            cached = copy.deepcopy(self.answer_cache[cache_key])
            cached.setdefault("meta", {})["cached"] = True
            cached["meta"]["latency_ms"] = 0
            session["last_objects"] = cached.get("objects") or []
            if cached.get("objects") and cached.get("meta", {}).get("route") == "ground":
                session["last_subject"] = cached["objects"][0]
            session["last_query"] = query
            hist = list(payload.get("messages") or [])[-11:]
            hist.append({"role": "assistant", "content": cached.get("reply") or ""})
            session["history"] = hist[-12:]
            session["updated"] = time.time()
            self._save(sid, session)
            if emit:
                emit({"type": "cache", "text": "命中同图同问题缓存"})
                if cached.get("objects"):
                    emit({"type": "boxes", "tool": cached.get("meta", {}).get("route"), "objects": cached["objects"]})
                emit({"type": "final", **cached})
            print(f"[agent] cache_hit key={cache_key[1][:30]!r}", flush=True)
            return cached

        trace: list[dict] = []
        timings: dict = {}
        # Perception（YOLOE NPU + ViT/ConvNeXt GPU）与 Qwen Planner 并行
        from concurrent.futures import ThreadPoolExecutor

        def _perc():
            t0 = time.perf_counter()
            self.rt.index_media(media)
            return int((time.perf_counter() - t0) * 1000)

        perception_pool = ThreadPoolExecutor(max_workers=1)
        perception_future = perception_pool.submit(_perc)
        parts = self._split(query)
        t_plan0 = time.perf_counter()
        steps = [self._make_step(p, i, session, sample) for i, p in enumerate(parts)]
        timings["plan_ms"] = int((time.perf_counter() - t_plan0) * 1000)
        try:
            timings["perception_ms"] = perception_future.result(timeout=300)
        except Exception as exc:
            timings["perception_ms"] = None
            import traceback
            print(f"[agent] perception failed: {exc}", flush=True)
            print(traceback.format_exc()[-1500:], flush=True)
        finally:
            perception_pool.shutdown(wait=False)
        if not steps:
            steps = [{"tool": "chat", "args": {}, "part": query, "chosen_by": "rule"}]
        trace.append({"type": "thought", "text": self._thought(steps)})
        trace.append({
            "type": "plan",
            "steps": [{"tool": s["tool"], "args": s["args"], "chosen_by": s["chosen_by"]} for s in steps],
        })
        self._emit(emit, trace[0])
        self._emit(emit, trace[1])

        objects: list[dict] = []
        observations: list[dict] = []
        tool_times: list[dict] = []
        prior_objects: list[dict] = list(session.get("last_objects") or [])

        for step in steps:
            self._emit(emit, {"type": "tool_start", "tool": step["tool"], "args": step["args"]})
            before = len(trace)
            result = self._execute(step, image, sample, session, prior_objects, trace)
            for event in trace[before:]:
                self._emit(emit, event)
            if result.get("objects"):
                self._emit(emit, {"type": "boxes", "tool": result.get("tool"), "objects": result["objects"]})
            print(
                f"[agent] tool={result.get('tool')} status={result.get('status')} "
                f"ms={result.get('duration_ms')} objects={len(result.get('objects') or [])}",
                flush=True,
            )
            tool_times.append({"tool": result.get("tool"), "ms": result.get("duration_ms")})
            observations.append(result)
            if result.get("facts"):
                session["event_facts"] = result["facts"]     # 供后续自由对话引用
            if result.get("objects"):
                objects = result["objects"]
                prior_objects = result["objects"]
                if result.get("tool") == "ground":
                    session["last_subject"] = objects[0]

        t_comp0 = time.perf_counter()
        reply = self._final(query, observations, objects, sample, media, session, trace, emit)
        timings["compose_ms"] = int((time.perf_counter() - t_comp0) * 1000)
        timings["tools"] = tool_times
        timings["total_ms"] = int((time.perf_counter() - started) * 1000)
        evidence = next((o.get("evidence") for o in observations if o.get("evidence")), None)
        call = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "query": query,
                "tools": [s["tool"] for s in steps], "answer": reply, "evidence": evidence}
        hist_calls = list(session.get("agent_history") or [])
        hist_calls.append(call)
        session["agent_history"] = hist_calls[-20:]
        session["last_objects"] = objects or prior_objects
        session["last_query"] = query
        hist = list(payload.get("messages") or [])[-11:]
        hist.append({"role": "assistant", "content": reply})
        session["history"] = hist[-12:]
        session["updated"] = time.time()
        self._save(sid, session)
        self._gc()

        result = {
            "reply": reply,
            "objects": objects,
            "trace": trace,
            "evidence": evidence,
            "history": session.get("agent_history")[-10:],
            "meta": {
                "engine": "rexseek",
                "model": "RexSeek-3B + Qwen2.5-1.5B",
                "demo": False,
                "route": steps[0]["tool"] if len(steps) == 1 else "multi-step",
                "device": self.rt.device,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "steps": len(steps),
                "session": sid,
                "timings": timings,
            },
        }
        print(f"[agent] done total_ms={timings['total_ms']} plan={timings['plan_ms']} perc={timings['perception_ms']} "
              f"tools={[(t['tool'], t['ms']) for t in tool_times]} compose={timings['compose_ms']}", flush=True)
        if cache_key:
            if len(self.answer_cache) >= 128:
                self.answer_cache.pop(next(iter(self.answer_cache)))
            stored = copy.deepcopy(result)
            stored.setdefault("meta", {})["cached"] = False
            self.answer_cache[cache_key] = stored
        self._emit(emit, {"type": "final", **result})
        return result

    # ----------------------------------------------------------------- 规划
    @staticmethod
    def _answer_key(image_key, query, event_id=None):
        # 事件上下文下的回答依赖台账实时状态（进行中/已结束、人数、安全帽），不能跨轮缓存
        if event_id:
            return None
        if not image_key:
            return None
        q = norm(query)
        if not q:
            return None
        # 指代 / 相对位置 / 多步追问不能跨轮缓存
        if any(w in q for w in ("她", "他", "它", "刚才", "上一个", "第二个", "第三个",
                                 "这个", "那个", "再", "然后", "接着", "左边", "右边",
                                 "上面", "下面", "旁边")):
            return None
        return (image_key, q)

    @staticmethod
    def _last_user(payload):
        for msg in reversed(payload.get("messages") or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    @staticmethod
    def _split(query: str) -> list[str]:
        if any(w in norm(query) for w in GREET + THANKS):
            return [query]
        # 显式连接词
        parts = re.split(
            r"(?:[，,；;]\s*)?(?:然后|接着|再|并且|同时|最后|接下来|随后|，并)\s*",
            query,
        )
        # 逗号 + 指代/方位/序数，也视为新子任务（“找出A，她右边是谁”）
        expanded: list[str] = []
        for part in parts:
            chunks = re.split(r"[，,]\s*(?=(?:她|他|它|右边|左边|上面|下面|前方|后方|第二|第三|最后))", part)
            expanded.extend(c for c in chunks if c.strip(" ，,；;。"))
        parts = expanded or parts
        parts = [p.strip(" ，,；;。") for p in parts if p.strip(" ，,；;。")]
        # 规划上限 3 步，避免一次请求无限展开
        return parts[:3]

    def _guardrail(self, part: str, session: dict | None) -> str | None:
        """只保留"必须由规则兜住"的场景：台账、装备、事件统计、区域归属。"""
        q = norm(part)
        if any(w in q for w in ("几次", "几起", "多少起", "台账", "历史", "今天有")):
            return "guard_events"
        if any(w in q for w in ("安全帽", "头盔", "反光背心", "口罩")):
            return "ppe_check"
        has_event = bool((session or {}).get("event_id"))
        if has_event and not any(w in q for w in ("在哪", "哪里", "位置")):
            if any(w in q for w in ("几点", "多久", "持续", "期间", "一共", "几个人", "多少人",
                                    "开始", "结束", "概况", "情况", "总结", "风险")):
                return "guard_segment"
        if any(w in q for w in ("危险区", "禁区", "值守区")) and any(
                w in q for w in ("进入", "进去", "在内", "在不在", "是否在", "有没有进", "有没有人", "有人吗")):
            return "zone_query"
        return None

    def _make_step(self, part: str, index: int, session: dict, sample: dict | None) -> dict:
        # ① 规则护栏（安全/确定性优先）
        guard = self._guardrail(part, session)
        if guard:
            return {"tool": guard, "args": self._rule_args(guard, part), "part": part, "chosen_by": "guardrail"}

        # ② Agent 决策：交给 Qwen，带完整工具目录和上下文
        ctx = []
        if session.get("event_id"):
            ctx.append(f"【事件】事件 #{session.get('event_id')}")
        kind = (sample or {}).get("kind") or "图片"
        ctx.append(f"【画面】{kind}")
        if session.get("last_subject"):
            ctx.append(f"【上文主语】{session['last_subject'].get('label')}")
        call = None
        try:
            call = self.rt.tool_call(part, "；".join(ctx))
        except Exception:
            call = None
        if call and self._valid_call(call.get("tool"), call.get("args") or {}, part):
            tool = call["tool"]
            return {"tool": tool, "args": self._norm_args(tool, call.get("args") or {}, part),
                    "part": part, "chosen_by": "agent"}

        # ③ 兜底：退回关键词规则
        rule = self._rule_tool(part, index, session)
        if rule:
            return {"tool": rule, "args": self._rule_args(rule, part), "part": part, "chosen_by": "rule"}
        return {"tool": "ground", "args": {"target": self._clean_target(strip_question(part))},
                "part": part, "chosen_by": "fallback"}

    def _valid_call(self, tool, args: dict, part: str) -> bool:
        """校验 Qwen 返回的 tool call，拒绝明显不合法的工具/参数。"""
        if tool not in ("ground", "count", "describe", "relative", "resolve", "clarify", "chat",
                        "guard_check", "guard_events", "guard_segment", "zone_query", "ppe_check", "attribute"):
            return False
        q = norm(part)
        if tool == "resolve":
            # resolve 必须真的有指代/序数，不能把“求婚的是谁”当指代
            return any(w in q for w in ("她", "他", "它", "刚才", "上一个", "第二", "第三"))
        if tool == "relative":
            direction = str((args or {}).get("direction") or "").lower()
            return direction in ("left", "right", "above", "below") or bool(self._direction(part))
        if tool == "ground":
            return bool(str((args or {}).get("target") or "").strip() or strip_question(part))
        if tool == "count":
            return bool(str((args or {}).get("category") or "").strip() or strip_question(part))
        return True

    def _rule_tool(self, part: str, index: int, session: dict | None = None) -> str | None:
        q = norm(part)
        has_event = bool((session or {}).get("event_id"))
        # ---------- 意图优先的路由（不再"命中关键词即返回"） ----------
        # 1) 装备/属性：安全帽、头盔、反光背心 —— 必须走按帧检测，绝不能拿事件统计糊弄
        if any(w in q for w in ("安全帽", "头盔", "反光背心", "口罩", "护目镜")):
            return "ppe_check"
        # 2) 颜色/穿着（属性在答案里）—— 多假设定位
        if any(w in q for w in ("什么颜色", "穿什么", "穿着", "戴的什么", "什么衣服")):
            return "attribute"
        # 3) 位置/指代："在哪、哪个、是谁、找出" —— 指代定位，不查区域规则
        if any(w in q for w in ("在哪", "哪里", "哪个", "哪个是", "是谁", "找出", "定位", "长什么样")):
            return "ground"
        # 4) 区域归属：在不在危险区/值守区
        zone_words = any(w in q for w in ("危险区", "禁区", "值守区"))
        zone_ask = any(w in q for w in ("进入", "进去", "在内", "在区", "是否在", "有没有在", "有没有进",
                                        "有人吗", "有没有人", "有人在", "没人", "离岗", "在不在",
                                        "几个人", "多少人", "有人"))
        if zone_words and zone_ask:
            return "zone_query"
        # 5) 实时区域状态（现在/当前 + 区域）
        if zone_words and any(w in q for w in ("现在", "此刻", "当前")):
            return "guard_check"
        # 6) 台账统计
        if any(w in q for w in ("几次", "几起", "多少起", "台账", "历史", "今天有")):
            return "guard_events"
        # 7) 事件的时间/人数/片段统计
        if has_event and any(w in q for w in ("几点", "多久", "持续", "期间", "一共", "几个人", "多少人",
                                              "开始", "结束", "概况", "情况", "总结", "风险")):
            return "guard_segment"
        if has_event and any(w in q for w in GREET + THANKS):
            return None
        if has_event:
            return "guard_segment"
        if any(w in q for w in ("塔吊", "禁区", "吊装", "值守", "下面有人", "违规", "报警", "事件")):
            if any(w in q for w in ("几次", "几起", "多少起", "多少", "几点", "什么时候", "记录", "台账", "历史", "哪几")):
                return "guard_events"
            if any(w in q for w in ("为什么", "原因", "解释")):
                return "guard_explain"
            return "guard_check"
        if "安全帽" in q:
            return "ppe_check"
        if any(w in q for w in GREET + THANKS + CAP):
            return "chat"
        if any(w in q for w in DESCRIBE) and not self._direction(part):
            return "describe"
        if any(w in q for w in COUNT) or re.search(r"(几|多少)", q):
            return "count"
        if self._direction(part) and (
            any(w in q for w in ("她", "他", "它", "那个人", "这个", "刚才", "第")) or index > 0
        ):
            return "relative"
        if any(w in q for w in ("她", "他", "它", "刚才那个", "上一个", "第二个", "第三个")):
            return "resolve"
        if any(w in q for w in ("在哪", "哪里", "找出", "定位", "哪个", "哪位", "谁", "穿", "颜色", "背景", "目标", "人", "车", "番茄", "大巴", "摩托")):
            return "ground"
        return None

    def _rule_args(self, tool: str | None, part: str) -> dict:
        target = self._clean_target(strip_question(part))
        if tool == "ground":
            return {"target": target}
        if tool == "count":
            return {"category": target}
        if tool == "relative":
            return {"direction": self._direction(part) or "right", "target": target, "ordinal": self._ordinal(part)}
        if tool == "resolve":
            return {"phrase": target}
        if tool == "clarify":
            return {"question": "你指的是画面里的哪一个目标？"}
        if tool == "guard_check":
            return {"zone": DEFAULT_ZONE}
        if tool == "zone_query":
            return {"target": self._zone_target(part)}
        if tool == "ppe_check":
            return {"query": part}
        if tool == "vqa":
            return {"question": part}
        if tool == "attribute":
            return {"question": part}
        if tool == "ppe_check":
            return {"query": part}
        if tool in ("guard_events", "guard_explain"):
            return {}
        return {}

    def _norm_args(self, tool: str, args: dict, part: str) -> dict:
        target = self._clean_target(str(args.get("target") or args.get("category") or strip_question(part)))
        if tool == "ground":
            return {"target": target}
        if tool == "count":
            return {"category": self._clean_target(str(args.get("category") or strip_question(part)))}
        if tool == "relative":
            direction = str(args.get("direction") or "").lower()
            if direction not in ("left", "right", "above", "below"):
                direction = self._direction(part) or "right"
            try:
                ordinal = max(1, int(args.get("ordinal") or 1))
            except (TypeError, ValueError):
                ordinal = 1
            return {"direction": direction, "target": target, "ordinal": ordinal}
        if tool == "resolve":
            return {"phrase": str(args.get("phrase") or strip_question(part))}
        if tool == "clarify":
            return {"question": str(args.get("question") or "你指的是画面里的哪一个目标？")}
        if tool == "zone_query":
            return {"target": str(args.get("target") or self._zone_target(part))}
        if tool == "ppe_check":
            return {"query": str(args.get("query") or part)}
        if tool == "attribute":
            return {"question": str(args.get("question") or part)}
        return {}

    @staticmethod
    def _clean_target(text: str) -> str:
        t = re.sub(r"^(画面里|图片里|图里|照片里|一张|这个|那个|这|那|有|有没有|是不是|一[头只个辆颗张]?|的)+", "", (text or "").strip())
        return t.strip(" ，,；;。") or text

    @staticmethod
    def _ordinal(part: str) -> int:
        m = re.search(r"第([一二三四五六七八九十\d]+)", part)
        if not m:
            return 1
        raw = m.group(1)
        if raw.isdigit():
            return max(1, int(raw))
        table = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        return table.get(raw, 1)

    @staticmethod
    def _direction(part: str) -> str | None:
        q = norm(part)
        for zh, en in DIRS.items():
            if zh in q:
                return en
        return None

    @staticmethod
    def _thought(steps: list[dict]) -> str:
        if len(steps) == 1 and steps[0]["tool"] == "chat":
            return "这是一句闲聊或能力询问，不需要调用工具，直接回答即可。"
        names = {"ground": "定位目标", "count": "统计数量", "describe": "概述画面",
                 "relative": "结合上文目标看相对位置", "resolve": "解析指代", "clarify": "向用户追问", "chat": "直接回答",
                         "guard_segment": "整段片段抽帧分析"}
        if len(steps) == 1:
            s = steps[0]
            return f"用户在问「{s['part']}」，需要{names.get(s['tool'], s['tool'])}。"
        seq = "；".join(f"{i+1}) {names.get(s['tool'], s['tool'])}" for i, s in enumerate(steps))
        return f"这是一个需要 {len(steps)} 步的任务：{seq}。先拿到中间结果，再基于它回答。"

    # ----------------------------------------------------------------- 执行
    def _execute(self, step: dict, image, sample, session, prior_objects, trace) -> dict:
        tool = step["tool"]
        args = step["args"]
        t0 = time.perf_counter()
        result = {"tool": tool, "args": args, "status": "ok", "objects": []}
        try:
            if tool == "ground":
                result = self._tool_ground(image, sample, args["target"], trace)
            elif tool == "count":
                result = self._tool_count(image, sample, args["category"])
            elif tool == "guard_check":
                result = self._tool_guard_check(image, session)
            elif tool == "ppe_check":
                result = self._tool_ppe_check(image, sample, args.get("query") or query)
            elif tool == "guard_events":
                result = self._tool_guard_events(session)
            elif tool == "guard_segment":
                result = self._tool_guard_segment(session)
            elif tool == "zone_query":
                result = self._tool_zone_query(image, sample, args.get("target") or "人", trace)
            elif tool == "vqa":
                result = self._tool_vqa(image, sample, args.get("question") or query, trace)
            elif tool == "attribute":
                result = self._tool_attribute(image, sample, args.get("question") or query, trace)
            elif tool == "guard_explain":
                result = self._tool_guard_explain()
            elif tool == "describe":
                result = self._tool_describe(image, sample)
            elif tool == "relative":
                ref = session.get("last_subject")
                if not ref:
                    result = {"tool": "clarify", "args": {"question": "你指的是画面里的哪一个目标？先告诉我一个主体，我才能判断它的相对位置。"},
                              "status": "ok", "observation": "缺少上文主体，已转为追问", "objects": []}
                else:
                    result = self._tool_relative(image, sample, args["direction"], args["target"], ref, args.get("ordinal", 1))
            elif tool == "resolve":
                result = self._tool_resolve(args["phrase"], prior_objects, session)
            elif tool == "clarify":
                result = {"tool": tool, "args": args, "status": "ok",
                          "observation": args["question"], "objects": []}
            else:
                result = {"tool": "chat", "args": {}, "status": "ok", "observation": "", "objects": []}
        except Exception as exc:  # 工具异常：记录、重试一次、再降级
            trace.append({"type": "tool", "tool": tool, "args": args, "status": "error",
                          "observation": f"{type(exc).__name__}: {exc}", "duration_ms": int((time.perf_counter() - t0) * 1000), "objects": []})
            try:
                if tool == "ground":
                    result = self._tool_ground(image, sample, args["target"], trace, retry=True)
                else:
                    result = {"tool": tool, "args": args, "status": "error",
                              "observation": "工具执行失败，已记录异常", "objects": []}
            except Exception as exc2:
                result = {"tool": tool, "args": args, "status": "error",
                          "observation": f"重试仍失败：{exc2}", "objects": []}

        result["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        result.setdefault("observation", "")
        trace.append({"type": "tool", **result})
        return result

    def _tool_ground(self, image, sample, target, trace, retry=False) -> dict:
        intent = match_intent(sample, target)
        if intent:
            noun = intent.get("en_query") or self.rt.extract_target(target) or target
            ref = noun
            prompt = intent.get("gdino_prompt") or (sample or {}).get("gdino_prompt") or "person. object."
        else:
            # 两路：名词给 GroundingDINO/YOLOE 出候选框；完整描述给 RexSeek 选择目标
            noun = self.rt.extract_target(target) or target
            ref = self.rt.extract_refexp(target) or target
            prompt = (noun + ".") if noun and all(ord(c) < 128 for c in noun) else ((sample or {}).get("gdino_prompt") or "person. object.")
        threshold = (sample or {}).get("box_threshold", 0.2) if not retry else 0.15
        with self.rt.lock:
            boxes, scores, labels = self.rt.candidates(image, prompt, (sample or {}).get("roi"), threshold)
            if not self._compatible(noun, labels):
                boxes, scores, labels = boxes[:0], scores[:0], []
            raw = self.rt.ground(image, boxes, ref, 256) if len(boxes) else ""
        indices = []
        for tag in re.findall(r"<obj\d+>", raw):
            i = int(tag[4:-1])
            if i < len(boxes) and i not in indices:
                indices.append(i)
        objects = self._to_objects(boxes, scores, indices, image, (intent or {}).get("zh_label") or target)
        # 候选框会有大量重叠（YOLOE 同一目标出多框），按 IoU 去重，避免"找到 2 个目标"其实是同一个人
        if len(objects) > 1:
            def contained(a, b):
                # 一个框大部分被另一个框包住（人物全身 vs 上半身）也算重复
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                amin = max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
                return inter / amin > 0.75

            kept = []
            for o in objects:
                dup = False
                for k in kept:
                    if self._iou_box(o["box"], k["box"]) >= 0.5 or contained(o["box"], k["box"]):
                        dup = True
                        break
                if not dup:
                    kept.append(o)
            objects = kept
        if not objects:
            trace.append({"type": "recovery", "status": "warn",
                          "text": f"「{target}」没有匹配到候选目标；改用整图/统计兜底，不伪造框。"})
        return {"tool": "ground", "args": {"target": target}, "status": "ok" if objects else "empty",
                "observation": f"定位到 {len(objects)} 个目标" if objects else "没有定位到匹配目标",
                "objects": objects}

    def _tool_count(self, image, sample, category) -> dict:
        if "人" in category:
            en, display = "person", "人"
        elif "车" in category:
            en, display = "car", "车"
        elif "番茄" in category:
            en, display = "tomato", "番茄"
        else:
            en = self.rt.extract_target(category) or "object"
            display = en
        if en == "person":
            try:
                d = self._ppe_req("/api/ppe", {"image": self._image_data_url(image)})
                w, h = image.size
                ps = d.get("persons", [])
                objs = []
                for i, p in enumerate(ps):
                    b = p["person"]
                    norm = [max(0, min(1, b[0] / w)), max(0, min(1, b[1] / h)),
                            max(0, min(1, b[2] / w)), max(0, min(1, b[3] / h))]
                    objs.append({"id": f"person{i}", "label": f"人 {i+1}", "short": f"人 {i+1}",
                                 "color": PALETTE[i % len(PALETTE)], "box": [round(v, 5) for v in norm],
                                 "score": 1.0, "where": spatial(norm)})
                return {"tool": "count", "args": {"category": display}, "status": "ok",
                        "observation": f"统计到 {len(objs)} 个{display}", "objects": objs, "display": display}
            except Exception as exc:
                print(f"[agent] ppe person count fallback: {exc}", flush=True)
        with self.rt.lock:
            boxes, scores, labels = self.rt.candidates(image, en + ".", (sample or {}).get("roi"), (sample or {}).get("box_threshold", 0.2))
        if en == "person" and len(boxes):
            w, h = image.size
            keep = [i for i in range(len(boxes))
                    if (boxes[i][3] - boxes[i][1]) >= h * 0.18 and (boxes[i][2] - boxes[i][0]) >= w * 0.035]
            boxes, scores = boxes[keep], scores[keep]
            labels = [labels[i] for i in keep]
        objects = []
        for i in range(len(boxes)):
            b = boxes[i].tolist(); w, h = image.size
            norm = [max(0, min(1, b[0] / w)), max(0, min(1, b[1] / h)), max(0, min(1, b[2] / w)), max(0, min(1, b[3] / h))]
            objects.append({"id": f"obj{i}", "label": f"{display} {i+1}", "short": f"{display} {i+1}",
                            "color": PALETTE[i % len(PALETTE)], "box": [round(v, 5) for v in norm],
                            "score": round(float(scores[i]), 3), "where": spatial(norm)})
        return {"tool": "count", "args": {"category": display}, "status": "ok",
                "observation": f"统计到 {len(objects)} 个{display}", "objects": objects, "display": display}

    def _tool_describe(self, image, sample) -> dict:
        prompt = (sample or {}).get("gdino_prompt") or "person. object."
        with self.rt.lock:
            boxes, scores, labels = self.rt.candidates(image, prompt, (sample or {}).get("roi"), (sample or {}).get("box_threshold", 0.2))
        from collections import Counter
        stats = "、".join(f"{LABEL_ZH.get(k, k)}×{v}" for k, v in Counter(labels).most_common(8)) or "没有明显目标"
        return {"tool": "describe", "args": {}, "status": "ok", "observation": stats, "objects": [],
                "facts": {"画面中可识别的目标": stats}, "stats": stats}

    def _tool_relative(self, image, sample, direction, target, ref, ordinal=1) -> dict:
        w, h = image.size
        x1, y1, x2, y2 = ref["box"]
        if direction == "left":
            cx1, cx2 = max(0.0, x1 - 0.45), x1
        elif direction == "right":
            cx1, cx2 = x2, min(1.0, x2 + 0.45)
        elif direction == "above":
            cx1, cx2 = x1, x2
        else:
            cx1, cx2 = x1, x2
        if direction in ("left", "right"):
            cy1, cy2 = max(0.0, y1 - 0.15), min(1.0, y2 + 0.15)
        else:
            cy1, cy2 = (max(0.0, y1 - 0.5), y1) if direction == "above" else (y2, min(1.0, y2 + 0.5))
        crop = image.crop((int(cx1 * w), int(cy1 * h), int(cx2 * w), int(cy2 * h)))
        if crop.width < 40 or crop.height < 40:
            return {"tool": "relative", "args": {"direction": direction, "target": target},
                    "status": "empty", "observation": "参考目标另一侧没有足够区域", "objects": []}
        person_target = any(k in target for k in ("人", "谁", "他", "她")) or direction in ("left", "right")
        prompt = "person." if person_target else ((self.rt.extract_target(target) or "object") + ".")
        with self.rt.lock:
            boxes, scores, labels = self.rt.candidates(crop, prompt, None, (sample or {}).get("box_threshold", 0.2))
            en = "the person" if person_target else (self.rt.extract_target(target) or target)
            raw = self.rt.ground(crop, boxes, en, 128) if len(boxes) else ""
        indices = []
        for tag in re.findall(r"<obj\d+>", raw):
            i = int(tag[4:-1])
            if i < len(boxes) and i not in indices:
                indices.append(i)
        cw, ch = crop.size
        zh = {"left": "左", "right": "右", "above": "上", "below": "下", "side": "旁"}.get(direction, "")
        objects = []
        for i, idx in enumerate(indices):
            b = boxes[idx].tolist()
            nx1, ny1 = cx1 + b[0] / cw * (cx2 - cx1), cy1 + b[1] / ch * (cy2 - cy1)
            nx2, ny2 = cx1 + b[2] / cw * (cx2 - cx1), cy1 + b[3] / ch * (cy2 - cy1)
            norm = [nx1, ny1, nx2, ny2]
            label = f"{zh}侧的人"
            objects.append({"id": f"rel{idx}", "label": label, "short": label,
                            "color": PALETTE[i % len(PALETTE)], "box": [round(v, 5) for v in norm],
                            "score": round(float(scores[idx]), 3), "where": spatial(norm)})
        # 相对位置问题只保留离参考目标最近的一个，避免把背景人群全列出来
        if objects:
            ref_cx = (x1 + x2) / 2
            objects.sort(key=lambda o: abs((o["box"][0] + o["box"][2]) / 2 - ref_cx))
            idx = min(max(1, int(ordinal or 1)) - 1, len(objects) - 1)
            objects = objects[idx:idx + 1]
            if ordinal and int(ordinal) > 1:
                objects[0]["label"] = f"{zh}侧第{ordinal}个人"
                objects[0]["short"] = objects[0]["label"]
        return {"tool": "relative", "args": {"direction": direction, "target": target},
                "status": "ok" if objects else "empty",
                "observation": f"在参考目标{zh}侧找到 {len(objects)} 个目标" if objects else f"参考目标{zh}侧没有找到目标",
                "objects": objects, "reference": ref["label"], "direction": direction}

    def _tool_resolve(self, phrase, objects, session=None) -> dict:
        q = norm(phrase)
        objects = list(objects or [])
        primary = (session or {}).get("last_subject")
        if not objects and not primary:
            return {"tool": "resolve", "args": {"phrase": phrase}, "status": "error",
                    "observation": "上文没有可引用的目标", "objects": []}
        female = lambda o: any(k in o.get("label", "") for k in ("女", "裙", "女士"))
        male = lambda o: any(k in o.get("label", "") for k in ("男", "男士"))
        if any(w in q for w in ("她", "他", "它", "刚才", "上一个")) and not primary:
            return {"tool": "clarify", "args": {"phrase": phrase},
                    "status": "ok", "observation": "当前会话没有明确的主语", "objects": []}

        if "第二" in q and len(objects) > 1:
            obj = objects[1]
        elif "第三" in q and len(objects) > 2:
            obj = objects[2]
        elif "她" in q or "女" in q:
            if primary and female(primary):
                obj = primary
            else:
                obj = next((o for o in reversed(objects) if female(o)), objects[-1] if objects else primary)
        elif "他" in q or "男" in q:
            if primary and male(primary):
                obj = primary
            else:
                obj = next((o for o in reversed(objects) if male(o)), objects[-1] if objects else primary)
        else:
            obj = primary or objects[-1]
        return {"tool": "resolve", "args": {"phrase": phrase}, "status": "ok",
                "observation": f"引用上文目标：{obj['label']}（{obj['where']}）", "objects": [obj]}

    # ----------------------------------------------------------------- 收尾
    @staticmethod
    def _answer_from_facts(query: str, obs: dict) -> str | None:
        """事实型问题（时间/人数/安全帽）直接用检测结果拼答案：不经过小模型，保证数字不出错。"""
        f = obs.get("facts") or {}
        if not f:
            return None
        q = norm(query)
        enter = f.get("进入时间") or f.get("最近一起进入")
        leave = f.get("离开时间") or f.get("最近一起离开")
        dur = f.get("持续时长") or f.get("最近一起持续")
        total = f.get("期间去重总人数")
        if total is None:
            total = f.get("最近一起去重人数")
        ok, bad = f.get("佩戴安全帽人数"), f.get("未佩戴安全帽人数")   # 仅"用户主动问安全帽"时的按需检测结果
        parts = []

        if f.get("目标") is not None:
            tgt, n = f["目标"], f.get("定位到数") or 0
            if n == 0:
                return f"画面里没有定位到「{tgt}」。"
            bits = []
            din = f.get("危险区内数") or 0
            if any(w in q for w in ("危险区", "禁区", "进入", "进去")):
                if tgt in ("人", "人员", "画面里的人"):
                    bits.append(f"危险区内{'有 ' + str(din) + ' 人' if din else '没有人'}")
                else:
                    bits.append(f"「{tgt}」{'在' if din else '不在'}危险区内" + (f"（{din} 个）" if din else ""))
            if "值守区" in q:
                zd = f.get("值守区内数") or 0
                bits.append(f"值守区内{'有 ' + str(zd) + ' 人' if zd else '没有人'}")
            if bits:
                return "；".join(bits) + "。"

        if f.get("人员位置") and any(w in q for w in ("在哪", "哪里", "位置")):
            return f"画面里有 {f.get('画面内人员数')} 人，{f['人员位置']}。"

        ask_time = any(w in q for w in ("几点", "时间", "开始", "结束", "多久", "持续", "时候"))
        ask_people = any(w in q for w in ("几个人", "多少人", "人数", "其他", "还有", "几人", "多少位"))
        ask_helmet = any(w in q for w in ("安全帽", "头盔", "佩戴", "戴了", "戴没戴"))
        ask_appearance = any(w in q for w in ("穿", "颜色", "衣服", "长什么样", "长相", "表情", "发型",
                                              "手里", "拿着", "背包"))

        if ask_time and enter:
            if leave and leave != "仍在值守中":
                parts.append(f"{enter} 进入，{leave} 离开，持续 {dur}")
            else:
                parts.append(f"{enter} 进入，仍在值守中，已持续 {dur}")

        if ask_people:
            if total is not None:
                text = f"期间共 {total} 人进入禁区（跨帧去重）"
                if int(total) <= 1:
                    text += "，没有其他人进入"
                parts.append(text)
            elif f.get("当前禁区内人数") is not None:
                parts.append(f"当前禁区内 {f['当前禁区内人数']} 人")
            elif f.get("画面内人员数") is not None:
                parts.append(f"画面内 {f['画面内人员数']} 人")

        if ask_helmet:
            if ok is not None and total:
                if bad:
                    parts.append(f"安全帽 {ok}/{total} 人佩戴，{bad} 人未佩戴")
                else:
                    parts.append(f"安全帽全部佩戴（{ok}/{total}）")
            elif f.get("戴安全帽") is not None:
                parts.append(f"画面 {f.get('画面人数')} 人，戴安全帽 {f.get('戴安全帽')} 人、"
                             f"未戴 {f.get('未戴安全帽')} 人")


        if not parts and enter and any(w in q for w in ("风险", "情况", "怎么回事", "发生了什么",
                                                        "描述", "总结", "判断", "合规", "违规")):
            peak = f.get("同时在场的峰值人数") or f.get("最近一起最多人数") or 1
            kind = f.get("类型") or "危险区"
            if kind == "值守区":
                txt = f"这是一起值守区无人告警：{enter} 起无人值守"
                if leave and leave != "仍在值守中":
                    txt += f"，{leave} 恢复，持续 {dur}"
                return txt + "。需要确认值守岗位是否有人替代。"
            txt = f"这是一起危险区入侵事件：{enter} 进入"
            if leave and leave != "仍在值守中":
                txt += f"、{leave} 离开，持续 {dur}"
            txt += f"，最多 {peak} 人进入，已抓拍取证"
            return txt + "。属于违规进入危险区，需要现场核查。"

        if not parts:
            return None
        yes_no = (ask_helmet and not ask_people
                  and any(w in q for w in ("是否", "有没有", "戴没戴", "吗")))
        head = "是的，" if yes_no else ""
        return head + "；".join(parts) + "。"

    @staticmethod
    def _time_conflict(answer: str, facts: dict) -> bool:
        """答案里出现的事实中没有的时刻 = 模型编的，直接弃用。"""
        flat = json.dumps(facts, ensure_ascii=False)
        for t in re.findall(r"\b\d{2}:\d{2}:\d{2}\b", answer or ""):
            if t not in flat:
                return True
        return False

    def _compose(self, query: str, obs: dict, session: dict | None = None) -> str:
        """事实型问题走确定性答案；开放式问题交给对话模型（带事实约束），出错则回退。"""
        facts = obs.get("facts") or {}
        fallback = obs.get("reply") or obs.get("observation") or "任务完成。"
        direct = self._answer_from_facts(query, obs)
        if direct:
            return direct
        if not facts:
            return fallback
        lines = []
        if facts.get("事件概述"):
            lines.append(facts["事件概述"])
        lines += [f"- {k}：{v}" for k, v in facts.items()
                  if k != "事件概述" and v not in (None, "", [], {})]
        context = (
            "[工具事实·端侧抽帧检测结果，数字与时间必须照抄，不要编造]\n"
            + "\n".join(lines)
            + "\n[要求] 用 1~3 句中文直接回答用户这句话；只讲和问题相关的部分；"
              "不要罗列全部字段，不要说\"根据工具事实\"，不要出现事实里没有的时间、数字或外观细节。"
        )
        history = list((session or {}).get("history") or [])[-4:]
        history.append({"role": "user", "content": query})
        answer = ""
        try:
            with self.rt.lock:
                answer = self.rt.chat(history, context, 140)
        except Exception:  # noqa: BLE001
            answer = ""
        answer = (answer or "").strip()
        if len(answer) < 2 or self._time_conflict(answer, facts):
            return fallback
        return answer

    def _final(self, query, observations, objects, sample, media, session, trace, emit=None) -> str:
        if len(observations) == 1 and observations[0]["tool"] in ("vqa", "attribute"):
            return observations[0].get("reply") or observations[0].get("observation") or "没看清。"
        if observations and observations[0]["tool"] == "clarify":
            return observations[0].get("observation") or observations[0].get("args", {}).get("question", "你指的是哪一个目标？")
        if len(observations) == 1 and observations[0]["tool"] in ("guard_check", "ppe_check", "guard_events",
                                                                  "guard_explain", "guard_segment", "describe",
                                                                  "zone_query"):
            return self._compose(query, observations[0], session)
        if len(observations) == 1 and observations[0]["tool"] == "ground" and not objects:
            return "没有在画面里定位到匹配目标。"
        if len(observations) == 1 and observations[0]["tool"] == "ground" and objects:
            if len(objects) == 1:
                o = objects[0]
                return f"{o['label']}在{o['where']}。"
            return f"找到 {len(objects)} 个目标：" + "；".join(f"{o['label']}（{o['where']}）" for o in objects[:8]) + "。"
        if len(observations) > 1 and not any(o["tool"] == "chat" for o in observations):
            bits = []
            for i, o in enumerate(observations, 1):
                if o["tool"] == "count":
                    bits.append(f"第 {i} 步统计到 {len(o.get('objects', []))} 个{o.get('display', '目标')}")
                elif o["tool"] == "ground" and o.get("objects"):
                    bits.append(f"第 {i} 步定位到" + "、".join(f"{x['label']}（{x['where']}）" for x in o["objects"][:4]))
                elif o["tool"] == "relative" and o.get("objects"):
                    d = {"left": "左", "right": "右", "above": "上", "below": "下"}.get(o.get("direction"), "")
                    bits.append(f"第 {i} 步在{o.get('reference','参考目标')}{d}侧找到 1 个人（{o['objects'][0]['where']}）")
                else:
                    bits.append(f"第 {i} 步{o.get('observation','完成')}")
            return "；".join(bits) + "。"

        if observations and observations[0]["tool"] == "count":
            o = observations[0]
            return f"画面里一共有 {len(o.get('objects', []))} 个{o.get('display', '目标')}。"

        if len(observations) == 1 and observations[0]["tool"] == "relative" and objects:
            o = observations[0]
            d = {"left": "左", "right": "右", "above": "上", "below": "下"}.get(o.get("direction"), "")
            return f"{o.get('reference', '参考目标')}{d}侧找到 1 个人，位于{objects[0]['where']}。"

        if len(observations) == 1 and observations[0]["tool"] == "resolve" and objects:
            obj = objects[0]
            return f"你指的是{obj['label']}，它在{obj['where']}。"

        if any(o["tool"] == "chat" for o in observations) and not any(o.get("objects") for o in observations):
            context = self._media_context(sample, media)
            if session.get("event_facts"):
                context += "\n[当前查看的事件] " + "；".join(
                    f"{k}: {v}" for k, v in session["event_facts"].items() if v not in (None, "", [], {}))
            with self.rt.lock:
                reply = self.rt.chat(
                    (session.get("history") or []) + [{"role": "user", "content": query}],
                    context, 80, on_token=lambda t: self._emit(emit, {"type": "token", "text": t}),
                )
            return reply or self.rt._canned_chat(query)

        lines = []
        for i, o in enumerate(observations, 1):
            detail = o.get("observation") or ""
            if o["tool"] == "ground" and o.get("objects"):
                detail = "定位到 " + "、".join(f"{x['label']}（{x['where']}）" for x in o["objects"][:6])
            lines.append(f"{i}. {o['tool']}：{detail}")
        if objects:
            ctx = "；".join(f"{o['label']}在{o['where']}" for o in objects[:6])
        else:
            ctx = "没有定位到匹配目标"
        context = (
            f"{self._media_context(sample, media)}\n"
            f"[任务执行记录]\n" + "\n".join(lines) + "\n"
            f"[最终目标] {ctx}\n"
            "请用 1-3 句自然中文回答用户；多步任务要体现先后关系；不要编造执行记录之外的目标。"
        )
        with self.rt.lock:
            reply = self.rt.chat(
                (session.get("history") or []) + [{"role": "user", "content": query}],
                context, 96, on_token=lambda t: self._emit(emit, {"type": "token", "text": t}),
            )
        if not reply:
            reply = self.rt._template_reply(query, objects[0]["label"] if objects else query, objects, None)
        return reply

    @staticmethod
    def _emit(emit, event):
        if emit:
            try:
                emit(event)
            except Exception:
                pass

    @staticmethod
    def _media_context(sample, media):
        if sample:
            return f"[画面] {sample.get('title','未命名')}（{'视频' if sample.get('kind') == 'video' else '图片'}，{media.get('width')}×{media.get('height')}）"
        return f"[画面] 用户上传的{('视频' if media.get('kind') == 'video' else '图片')}（{media.get('width')}×{media.get('height')}）"

    @staticmethod
    def _compatible(query: str, labels) -> bool:
        q = (query or "").lower()
        labels = set(labels or [])
        if not labels:
            return False
        if os.environ.get("YOLOE_CLASS_AGNOSTIC") == "1":
            # YOLOE 只做候选框；类别由 RexSeek 按语言复核
            return True
        if any(k in q for k in ("person", "woman", "man", "people", "speaker", "人", "女士", "男士")):
            return "person" in labels
        if "tomato" in q or "番茄" in q:
            return bool(labels)
        if "bus" in q or "大巴" in q:
            return "bus" in labels
        if "car" in q or "汽车" in q:
            return "car" in labels or "truck" in labels
        if q.strip(" .") in ("animal", "animals", "wildlife"):
            return any(x in labels for x in ("dog", "cat", "bird", "horse", "elephant", "bear", "animal"))
        if "object" in q:
            return True
        return True

    def _image_data_url(self, image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=86)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _ppe_req(self, path, payload, timeout=90):
        req = urllib.request.Request(PPE_URL.rstrip("/") + path,
                                     data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    COLORS = [("红色", "person wearing red clothes"), ("蓝色", "person wearing blue clothes"),
              ("白色", "person wearing white clothes"), ("黑色", "person wearing black clothes"),
              ("黄色", "person wearing yellow clothes"), ("橙色", "person wearing orange clothes"),
              ("绿色", "person wearing green clothes"), ("灰色", "person wearing gray clothes")]

    @staticmethod
    def _pick_referent(persons, question):
        """按问题里的方位词/指代挑出"哪个人"。"""
        if not persons:
            return None
        q = norm(question)
        if any(w in q for w in ("左",)):
            return min(persons, key=lambda p: (p["box"][0] + p["box"][2]))
        if any(w in q for w in ("右",)):
            return max(persons, key=lambda p: (p["box"][0] + p["box"][2]))
        if "中间" in q or "中" in q:
            return min(persons, key=lambda p: abs((p["box"][0] + p["box"][2]) / 2 - 0.5))
        return max(persons, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))

    def _tool_attribute(self, image, sample, question, trace):
        """属性问题（穿什么颜色）用"多假设定位"回答：逐个假设找人，和指代目标的重合度最高者胜出。"""
        with self.rt.lock:
            boxes, scores, _labels = self.rt.candidates(image, "person.", None, 0.2)
        if len(boxes) == 0:
            return {"tool": "attribute", "args": {"question": question}, "status": "empty", "objects": [],
                    "observation": "画面里没有检测到人", "reply": "画面里没有检测到人。"}
        w, h = image.size
        px = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]
        norm_boxes = [[max(0, b[0] / w), max(0, b[1] / h), min(1, b[2] / w), min(1, b[3] / h)] for b in px]
        persons = [{"box": norm_boxes[i], "px": px[i]} for i in range(len(px))]
        ref = self._pick_referent(persons, question)
        if not ref:
            return {"tool": "attribute", "args": {"question": question}, "status": "empty", "objects": [],
                    "observation": "没有确定指代对象", "reply": "你说的是画面里的哪一个人？"}
        rx1, ry1, rx2, ry2 = ref["px"]

        def iou_px(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / (ua + 1e-9)

        def matched(phrase):
            try:
                with self.rt.lock:
                    raw = self.rt.ground(image, boxes, phrase, 128)
            except Exception:  # noqa: BLE001
                return []
            out = []
            for tag in re.findall(r"<obj\d+>", raw):
                i = int(tag[4:-1])
                if 0 <= i < len(px):
                    out.append(px[i])
            return out

        q = norm(question)
        where = ("左侧" if "左" in q else "右侧" if "右" in q else "中间" if ("中间" in q or "中" in q) else "")
        prefix = f"画面{where}的人" if where else "这个人"
        if "安全帽" in q or "头盔" in q:
            # 安全帽：用"戴安全帽的人"假设定位，看和指代目标是否重合
            hit = max((iou_px(ref["px"], b) for b in matched("person wearing a safety helmet")), default=0.0)
            worn = hit > 0.2
            reply = f"{prefix}{'戴' if worn else '没戴'}安全帽。"
            return {"tool": "attribute", "args": {"question": question}, "status": "ok", "objects": [],
                    "observation": reply, "reply": reply,
                    "facts": {"问题": question, "结论": reply}}

        scores_by_color = []
        for zh, en in self.COLORS:
            hits = matched(en)
            score = max((iou_px(ref["px"], b) for b in hits), default=0.0)
            scores_by_color.append((score, zh))
        scores_by_color.sort(reverse=True)
        best_score, best_color = scores_by_color[0]
        if best_score < 0.2:
            reply = f"{prefix}衣服颜色看不清。"
        else:
            reply = f"{prefix}穿{best_color}衣服。"
        return {"tool": "attribute", "args": {"question": question}, "status": "ok", "objects": [],
                "observation": reply, "reply": reply,
                "facts": {"问题": question, "结论": reply,
                          "候选": "，".join(f"{c}:{round(s, 2)}" for s, c in scores_by_color[:3])}}

    def _tool_vqa(self, image, sample, question, trace):
        """让 RexSeek 直接看图回答属性类问题（颜色/穿着/动作…）。"""
        try:
            with self.rt.lock:
                boxes, _scores, _labels = self.rt.candidates(image, "person. object.", None, 0.2)
                raw = self.rt.vqa(image, boxes, "看这张图回答问题：" + question + " 用一句简短中文回答，不要输出编号。", 96)
        except Exception as exc:  # noqa: BLE001
            return {"tool": "vqa", "args": {"question": question}, "status": "error", "objects": [],
                    "observation": str(exc), "reply": "画面细节没有解析出来。"}
        ans = re.sub(r"<[^>]+>", " ", raw or "")
        ans = re.sub(r"\s+", " ", ans).strip(" 。")
        if not ans:
            return {"tool": "vqa", "args": {"question": question}, "status": "empty", "objects": [],
                    "observation": "没有解析出答案", "reply": "这个细节没看清。"}
        return {"tool": "vqa", "args": {"question": question}, "status": "ok", "objects": [],
                "observation": ans, "reply": ans if ans.endswith("。") else ans + "。"}

    @staticmethod
    def _zone_target(part: str) -> str:
        text = str(part or "")
        m = re.search(r"(穿[^，。！？,.?!]{1,12}?的人|戴[^，。！？,.?!]{1,12}?的人|"
                      r"站[^，。！？,.?!]{1,12}?的人|[^，。！？,.?!]{1,8}?的人)", text)
        return m.group(1) if m else "人"

    @staticmethod
    def _iou_box(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / (ua + 1e-9)

    @staticmethod
    def _point_in_poly(x: float, y: float, poly) -> bool:
        if not poly or len(poly) < 3:
            return False
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _tool_zone_query(self, image, sample, target, trace):
        """开放描述定位 + 区域判定：如"穿蓝色衣服的人是否进入危险区"。"""
        g = self._tool_ground(image, sample, target, trace)
        objects = g.get("objects") or []
        try:
            with urllib.request.urlopen(PPE_URL.rstrip("/") + "/api/guard/zone", timeout=10) as resp:
                z = json.loads(resp.read().decode())
        except Exception:  # noqa: BLE001
            z = {}
        danger, duty = z.get("danger") or z.get("zone"), z.get("duty")
        for o in objects:
            b = o.get("box") or [0, 0, 0, 0]
            foot = ((b[0] + b[2]) / 2, b[3])          # 脚底（站位）
            mid = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)   # 框中心（坐着/被遮挡时更稳）
            o["in_danger"] = self._point_in_poly(*foot, danger) or self._point_in_poly(*mid, danger)
            o["in_duty"] = self._point_in_poly(*foot, duty) or self._point_in_poly(*mid, duty)
        in_danger = sum(1 for o in objects if o.get("in_danger"))
        in_duty = sum(1 for o in objects if o.get("in_duty"))
        if not objects:
            reply = f"画面里没有定位到「{target}」。"
        elif in_danger:
            reply = f"{target}：在危险区内，共 {in_danger} 个。"
        else:
            reply = f"{target}：不在危险区内。"
        if duty:
            reply += f"值守区内 {in_duty} 个。"
        return {"tool": "zone_query", "args": {"target": target}, "status": "ok" if objects else "empty",
                "observation": reply, "reply": reply, "objects": objects,
                "facts": {"目标": target, "定位到数": len(objects),
                          "危险区内数": in_danger, "值守区内数": in_duty}}

    def _tool_guard_check(self, image, session):
        t0 = time.perf_counter()
        d = self._ppe_req("/api/guard/frame", {
            "image": self._image_data_url(image), "zone": DEFAULT_ZONE,
            "session_id": session.get("_sid", "agent"), "ts": time.time(),
        })
        w, h = image.size
        always = []
        for i, p in enumerate(d.get("persons", [])):
            b = p["box"]
            norm = [max(0, min(1, b[0] / w)), max(0, min(1, b[1] / h)),
                    max(0, min(1, b[2] / w)), max(0, min(1, b[3] / h))]
            always.append({"id": f"person{i}", "label": "人员", "short": f"人员 {i+1}",
                           "color": PALETTE[i % len(PALETTE)], "box": [round(v, 5) for v in norm],
                           "score": round(float(p.get("score", 0)), 3), "where": spatial(norm),
                           "inside": bool(p.get("inside")), "status": "intrusion" if p.get("inside") else "person"})
        inside = int(d.get("inside") or 0)
        ep = d.get("episode") or d.get("event")
        if inside:
            reply = f"危险区内有 {inside} 人。"
            if ep and ep.get("enter_hms"):
                reply += f"本起事件 {ep['enter_hms']} 进入，已持续 {ep.get('duration_text') or '—'}。"
            if ep:
                reply += " 已抓拍并告警。"
        else:
            reply = "当前没有告警。"
        facts = {"当前危险区内人数": inside, "画面内人员数": len(always), "状态": d.get("status"),
                 "人员位置": "、".join(f"人员{i+1}在{o.get('where')}" for i, o in enumerate(always[:6])) or None}
        if ep:
            facts.update({"当前事件进入时间": ep.get("enter_hms"), "当前事件已持续": ep.get("duration_text"),
                          "当前事件最多人数": ep.get("persons_max")})
        return {"tool": "guard_check", "args": {"zone": DEFAULT_ZONE}, "status": "ok",
                "observation": f"persons={len(always)} inside={inside} status={d.get('status')}",
                "reply": reply, "objects": always, "facts": facts,
                "evidence": d.get("evidence") or (d.get("event") or {}).get("evidence")}

    def _tool_ppe_check(self, image, sample, query):
        d = self._ppe_req("/api/mission", {
            "mission": "ppe_helmet", "query": query or "找出没戴安全帽的人",
            "image": self._image_data_url(image),
        })
        w, h = image.size
        objects = []
        for i, p in enumerate(d.get("findings", [])):
            b = p.get("person") or [0, 0, 0, 0]
            norm = [max(0, min(1, b[0] / w)), max(0, min(1, b[1] / h)),
                    max(0, min(1, b[2] / w)), max(0, min(1, b[3] / h))]
            objects.append({"id": f"ppe{i}", "label": "未佩戴安全帽" if p.get("status") == "not_worn" else "已佩戴安全帽",
                            "short": "安全帽", "color": PALETTE[i % len(PALETTE)],
                            "box": [round(v, 5) for v in norm], "score": p.get("confidence", 0.5),
                            "where": spatial(norm), "status": p.get("status")})
        sm = d.get("summary") or {}
        facts = {"画面人数": sm.get("persons"), "戴安全帽": sm.get("worn"),
                 "未戴安全帽": sm.get("not_worn"), "检测到安全帽数": sm.get("helmets")}
        return {"tool": "ppe_check", "args": {"query": query}, "status": "ok",
                "observation": d.get("answer") or "安全帽检查完成",
                "reply": d.get("answer") or "安全帽检查完成", "facts": facts,
                "objects": objects, "evidence": d.get("evidence"), "summary": sm}

    @staticmethod
    def _episode_text(ev: dict) -> str:
        enter = ev.get("enter_hms") or ""
        exit_ = ev.get("exit_hms")
        dur = ev.get("duration_text") or "-"
        persons = ev.get("persons_max") or 0
        if exit_:
            return f"{enter} 进入、{exit_} 离开，持续 {dur}，最多 {persons} 人"
        return f"{enter} 进入，值守中（已持续 {dur}，最多 {persons} 人）"

    def _tool_guard_segment(self, session=None):
        """整段事件分析：起止时间 + 跨帧去重人数 + 安全帽佩戴。"""
        event_id = (session or {}).get("event_id")
        try:
            if not event_id:
                with urllib.request.urlopen(PPE_URL.rstrip("/") + "/api/guard/events", timeout=20) as resp:
                    evs = (json.loads(resp.read().decode())).get("events") or []
                if not evs:
                    return {"tool": "guard_segment", "args": {}, "status": "ok", "objects": [],
                            "observation": "没有事件", "reply": "今天还没有区域告警事件。"}
                event_id = evs[0]["id"]
            with urllib.request.urlopen(PPE_URL.rstrip("/") + f"/api/guard/segment/{int(event_id)}",
                                        timeout=40) as resp:
                seg = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            return {"tool": "guard_segment", "args": {"event_id": event_id}, "status": "error", "objects": [],
                    "observation": str(exc), "reply": "片段分析暂时不可用，请稍后再试。"}
        ev = seg.get("event") or {}
        sm = seg.get("summary") or {}
        frames = seg.get("frames") or []
        total = sm.get("persons_total")
        helmet_ok = sm.get("helmet_ok")
        span = self._episode_text(ev) if ev else "片段"
        reply = f"这一段（事件 #{ev.get('id')}）：{span}；共抽帧 {len(frames)} 张做整段分析。"
        if total is None:
            reply += "片段还在分析中，稍后再问一次即可。"
        else:
            reply += f"期间共 {total} 人（同一个人跨帧已合并，不会重复计数）"
            if helmet_ok is not None:
                if helmet_ok >= total:
                    reply += f"，全部佩戴安全帽（{helmet_ok}/{total}）。"
                else:
                    reply += f"，其中 {total - helmet_ok} 人未佩戴安全帽（佩戴 {helmet_ok}/{total}）。"
            else:
                reply += "。"
        span_txt = (f"{ev.get('enter_hms')} 进入，{ev.get('exit_hms')} 离开，持续 {ev.get('duration_text')}"
                    if ev.get("exit_hms") else
                    f"{ev.get('enter_hms')} 进入，仍在值守中，已持续 {ev.get('duration_text')}")
        people_txt = (f"期间进入禁区 {total} 人（跨帧去重，同一个人不重复计数）"
                      + ("，没有其他人进入" if (total is not None and total <= 1) else "")
                      + f"；同时在场的峰值 {sm.get('peak') or ev.get('persons_max')} 人")
        helmet_txt = ""
        facts = {
            "事件概述": (f"事件 #{ev.get('id')}（{ev.get('kind_label') or '危险区'}）：" + span_txt + "。"
                       + people_txt + f"。抽帧 {len(frames)} 张。"),
            "类型": ev.get("kind_label") or "危险区",
            "事件编号": ev.get("id"),
            "进入时间": ev.get("enter_hms"),
            "离开时间": ev.get("exit_hms") or "仍在值守中",
            "持续时长": ev.get("duration_text"),
            "同时在场的峰值人数": sm.get("peak") or ev.get("persons_max"),
            "抽帧数": len(frames),
            "期间去重总人数": total,
            "期间进入人数": (None if total is None else f"{total} 人" + ("（没有其他人进入）" if total <= 1 else "")),
        }
        if sm.get("tracks"):
            facts["人员出现区间"] = [
                f"#{t.get('id')} {datetime.fromtimestamp(t['first_ts']).strftime('%H:%M:%S')}"
                f"-{datetime.fromtimestamp(t['last_ts']).strftime('%H:%M:%S')}"
                f"（{'戴' if t.get('helmet') else '未戴'}安全帽）"
                for t in sm["tracks"][:6]
            ]
        return {"tool": "guard_segment", "args": {"event_id": ev.get("id")}, "status": "ok",
                "observation": reply, "reply": reply, "objects": [], "facts": facts,
                "evidence": ev.get("evidence"), "frames": frames[:12], "summary": sm}

    def _tool_guard_events(self, session=None):
        event_id = (session or {}).get("event_id")
        total = 0
        if event_id:
            # 台账点进来的这一起：优先按事件 id 回答
            try:
                with urllib.request.urlopen(PPE_URL.rstrip("/") + f"/api/guard/event/{int(event_id)}",
                                            timeout=20) as resp:
                    d = json.loads(resp.read().decode())
                ev = d.get("event")
                total = d.get("total", 0)
                if ev:
                    reply = (f"你正在查看的事件 #{ev.get('id')}：{self._episode_text(ev)}。"
                             f"今天共 {total} 起区域告警事件（一次告警记一条）。")
                    return {"tool": "guard_events", "args": {"event_id": ev.get("id")}, "status": "ok",
                            "observation": reply, "reply": reply, "objects": [],
                            "evidence": ev.get("evidence"), "events": [ev]}
            except Exception:  # noqa: BLE001
                pass
        with urllib.request.urlopen(PPE_URL.rstrip("/") + "/api/guard/events", timeout=20) as resp:
            d = json.loads(resp.read().decode())
        events = d.get("events", [])
        total = d.get("total", len(events))
        if events:
            latest = events[0]
            reply = (f"今天共 {total} 起区域告警事件（一次告警记一条）。"
                     f"最近一起：{self._episode_text(latest)}。")
            evidence = latest.get("evidence")
        else:
            reply, evidence = "今天还没有区域告警事件。", None
        facts = {"话术": "一次进入记一条事件", "今天事件总数": total}
        if events:
            latest = events[0]
            facts.update({"最近一起进入": latest.get("enter_hms"),
                          "最近一起离开": latest.get("exit_hms") or "仍在值守中",
                          "最近一起持续": latest.get("duration_text"),
                          "最近一起最多人数": latest.get("persons_max"),
                          "最近一起去重人数": latest.get("persons_total") or None,
                          "最近一起类型": latest.get("kind_label")})
        return {"tool": "guard_events", "args": {}, "status": "ok", "observation": reply,
                "reply": reply, "objects": [], "facts": facts, "evidence": evidence, "events": events[:10]}

    def _tool_guard_explain(self):
        d = self._ppe_req("/api/guard/ask", {"question": "为什么刚才报警？"})
        return {"tool": "guard_explain", "args": {}, "status": "ok",
                "observation": d.get("answer", ""), "reply": d.get("answer", ""),
                "objects": [], "evidence": d.get("evidence")}

    def _to_objects(self, boxes, scores, indices, image, label):
        w, h = image.size
        out = []
        for i, idx in enumerate(indices):
            b = boxes[idx].tolist()
            norm = [max(0, min(1, b[0] / w)), max(0, min(1, b[1] / h)), max(0, min(1, b[2] / w)), max(0, min(1, b[3] / h))]
            lab = label if len(indices) == 1 else f"{label} {i+1}"
            out.append({"id": f"obj{idx}", "label": lab, "short": lab[:10] if len(lab) > 10 else lab,
                        "color": PALETTE[i % len(PALETTE)], "box": [round(v, 5) for v in norm],
                        "score": round(float(scores[idx]), 3), "where": spatial(norm)})
        return out

    # ----------------------------------------------------------------- 会话
    def _session(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if not s and self.store:
            try:
                s = self.store.load(sid)
            except Exception:
                s = None
        if not s:
            s = {"last_objects": [], "last_subject": None, "last_query": "",
                 "history": [], "image_key": None, "updated": time.time()}
        for key, default in (("last_objects", []), ("last_subject", None), ("last_query", ""),
                             ("history", []), ("image_key", None)):
            s.setdefault(key, default)
        self.sessions[sid] = s
        return s

    def _save(self, sid: str, session: dict) -> None:
        if not self.store:
            return
        try:
            self.store.save(sid, session)
        except Exception as exc:
            print(f"[agent] session save failed: {exc}", flush=True)

    def _gc(self):
        now = time.time()
        for k in [k for k, v in self.sessions.items() if now - v.get("updated", now) > 7200]:
            self.sessions.pop(k, None)
