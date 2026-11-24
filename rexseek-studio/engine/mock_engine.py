"""演示引擎：基于内置样例的知识库做意图路由。

它不加载任何模型，但严格复用和真实后端一致的返回结构，
方便后续把 ``--engine`` 切到 proxy 而不动前端。
"""

from __future__ import annotations

import re
from typing import Any

_BROAD = ("描述", "介绍", "看看", "是什么", "有什么", "都有", "整体", "场景")
_GREETING = ("你好", "您好", "hi", "hello", "嗨", "在吗")


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[\s，。！？、,.!?~…]+", "", text)
    return text


def _last_user_text(payload: dict) -> str:
    messages = payload.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


class MockEngine:
    name = "mock"
    model = "RexSeek-3B · demo"

    def __init__(self, catalog: dict):
        self.catalog = catalog
        self.samples = {s["id"]: s for s in catalog.get("samples", [])}

    # ------------------------------------------------------------------ api
    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "demo": True,
            "note": "内置样例 + 规则路由；切换 --engine proxy 可接真实 RexSeek 推理服务",
        }

    def respond(self, payload: dict) -> dict:
        query = _last_user_text(payload)
        media = payload.get("media") or {}
        sample = self.samples.get(media.get("sample_id") or "")

        if not query:
            return self._result("想让我看画面里的什么？可以直接问我某个物体在哪，或者让我数一数。")

        if _norm(query) in {_norm(g) for g in _GREETING}:
            return self._result(
                "你好，我是 RexSeek。左侧的图片或视频已经就绪，你可以用自然语言指代画面里的任意对象，"
                "我会把对应的区域框出来。"
            )

        if sample is None:
            return self._unknown(media, query)

        intent = self._match(sample, query)
        if intent:
            return self._from_intent(sample, intent)

        return self._fallback(sample, query)

    # ------------------------------------------------------------- internal
    def _match(self, sample: dict, query: str) -> dict | None:
        q = _norm(query)
        best: dict | None = None
        best_score = 0
        for intent in sample.get("intents", []):
            score = 0
            for kw in intent.get("keywords", []):
                k = _norm(kw)
                if k and k in q:
                    score += max(2, len(k))
            if score > best_score:
                best, best_score = intent, score
        return best

    def _from_intent(self, sample: dict, intent: dict) -> dict:
        ids = intent.get("object_ids") or []
        count = intent.get("count_value", len(ids))
        reply = (intent.get("reply") or "").replace("{count}", str(count))
        objects = self._pick(sample, ids)
        return self._result(reply, objects, intent.get("id"))

    def _fallback(self, sample: dict, query: str) -> dict:
        q = _norm(query)
        # 直接命中对象名称（例如 "Alvin KWOCK"）时也给出框
        hits = []
        for obj in sample.get("objects", []):
            label = _norm(obj.get("label", ""))
            short = _norm(obj.get("short", ""))
            if (label and label in q) or (short and short in q):
                hits.append(obj["id"])
        if hits:
            return self._result("找到了，我已经把对应的区域框出来了。", self._pick(sample, hits), "label-hit")

        reply = sample.get("default_reply") or "我在画面里看到了一些对象，可以告诉我你想定位哪一个。"
        if any(word in q for word in _BROAD):
            objects = self._pick(sample, [o["id"] for o in sample.get("objects", [])][:3])
            return self._result(reply, objects, "overview")
        return self._result(reply, [], "fallback")

    def _pick(self, sample: dict, ids: list[str]) -> list[dict]:
        by_id = {o["id"]: o for o in sample.get("objects", [])}
        result = []
        for oid in ids:
            obj = by_id.get(oid)
            if not obj:
                continue
            item = {
                "id": obj["id"],
                "label": obj["label"],
                "short": obj.get("short") or obj["label"],
                "color": obj.get("color", "#5eead4"),
                "box": obj["box"],
                "score": obj.get("score", 0.9),
                "where": obj.get("where", ""),
            }
            if obj.get("track"):
                item["track"] = obj["track"]
            result.append(item)
        return result

    def _unknown(self, media: dict, query: str) -> dict:
        kind = media.get("kind") or "image"
        name = media.get("name") or ("这个视频" if kind == "video" else "这张图片")
        reply = (
            f"「{name}」已经载入。当前是**演示模式**，只有内置示例素材接入了定位知识库，"
            "自定义素材需要切到真实 RexSeek 后端才能理解。\n\n"
            "你可以点左下角「示例素材」里的任意一张，直接体验提问 → 框选的完整链路。"
        )
        if any(w in _norm(query) for w in ("多大", "尺寸", "分辨率", "大小")):
            w, h = media.get("width"), media.get("height")
            if w and h:
                reply = f"这张画面是 {w} × {h} 像素，宽高比约 {w / h:.2f} : 1。"
        return self._result(reply, [])

    def _result(self, reply: str, objects: list[dict] | None = None, intent: str | None = None) -> dict:
        return {
            "reply": reply,
            "objects": objects or [],
            "meta": {
                "engine": self.name,
                "model": self.model,
                "demo": True,
                "intent": intent,
            },
        }
