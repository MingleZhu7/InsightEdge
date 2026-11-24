"""把对话请求转发到一个真实的 RexSeek 推理服务。

约定（详见 README「接入真实后端」）:

    POST {endpoint}/api/chat
    Content-Type: application/json

    {
      "messages": [{"role": "user", "content": "穿蓝裙子的女士在哪？"}],
      "media": {
        "sample_id": "proposal",      # 或 null
        "kind": "image",
        "name": "demo1.jpg",
        "width": 1000,
        "height": 1504,
        "data_url": "data:image/jpeg;base64,..."   # 可选，自定义上传时携带
      },
      "video_time": 0.0
    }

返回（字段与本地 mock 完全一致）:

    {
      "reply": "...",
      "objects": [
        {"id": "...", "label": "...", "short": "...", "color": "#35e0ff",
         "box": [x1, y1, x2, y2], "score": 0.96, "where": "...",
         "track": [{"t": 0.0, "box": [...]}]}
      ],
      "meta": {"engine": "rexseek", "model": "RexSeek-3B", "latency_ms": 812}
    }
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class ProxyEngine:
    name = "proxy"
    model = "RexSeek-3B"

    def __init__(self, endpoint: str, timeout: float = 180):
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/api/chat"):
            self.base = endpoint[: -len("/api/chat")]
        else:
            self.base = endpoint
        self.url = self.base + "/api/chat"
        self.timeout = timeout
        self.public_endpoint = endpoint

    def describe(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "demo": False,
            "endpoint": self.public_endpoint,
        }

    def create_session(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/api/session",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"session service {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 session 服务: {exc.reason}") from exc

    def get_session(self, sid: str) -> dict:
        req = urllib.request.Request(self.base + "/api/session/" + urllib.parse.quote(sid), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"session_id": sid, "exists": False}
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"session service {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 session 服务: {exc.reason}") from exc

    def index(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/api/index",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"索引服务返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接索引服务: {exc.reason}") from exc

    def open_stream(self, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/api/chat/stream",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"推理服务返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接推理服务: {exc.reason}") from exc

    def respond(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"推理服务返回 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接推理服务 {self.url}: {exc.reason}") from exc

        if not isinstance(data, dict) or "reply" not in data:
            raise RuntimeError("推理服务返回格式不符合约定（缺少 reply 字段）")

        data.setdefault("objects", [])
        meta = data.setdefault("meta", {})
        meta.setdefault("engine", "rexseek")
        meta.setdefault("model", self.model)
        meta.setdefault("demo", False)
        return data
