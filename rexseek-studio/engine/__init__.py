"""推理引擎。

- ``mock``：内置样例 + 规则路由，无需 GPU，用于演示交互链路。
- ``proxy``：把请求转发给一个兼容的 RexSeek 推理服务（自带 GPU 的机器）。

真实后端的接口约定见 README.md 的「接入真实后端」一节。
"""

from .mock_engine import MockEngine
from .proxy_engine import ProxyEngine


def create_engine(kind: str, catalog: dict, **options):
    kind = (kind or "mock").lower()
    if kind == "mock":
        return MockEngine(catalog)
    if kind == "proxy":
        endpoint = options.get("endpoint")
        if not endpoint:
            raise SystemExit("--engine proxy 需要同时提供 --endpoint")
        return ProxyEngine(endpoint, timeout=options.get("timeout", 180))
    raise SystemExit(f"未知引擎: {kind}（可选: mock / proxy）")
