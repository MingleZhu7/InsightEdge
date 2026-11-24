# 演示素材（已随仓库）

`web/assets/samples/` 与 `web/media/` 已入库，clone 后即可直接演示。

## 通用演示（8849）

| 文件 | 对应示例 | 尺寸 |
|---|---|---|
| `demo9.jpg` | tomatoes 番茄成熟度 | 1083×1508 |
| `demo1.jpg` | proposal 桥上的求婚 | |
| `demo3.jpg` | webinar 讲者海报 | |
| `demo6.jpg` | team 团队合影 | |
| `helmet_all.jpg` / `helmet_mixed.jpg` | PPE 演示 | |

元数据（标题、建议问法、预置框）在 `data/samples_general.json`。

## 值守台（8848）

| 文件 | 说明 |
|---|---|
| `site_demo.mp4` / `site_demo.jpg` | 默认演示视频（1280×720，约 10 s）与封面 |
| `site_20261122_*.mp4` | 另两段现场素材（会在左侧素材区列出） |

第一次上传视频时，前端会把文件落到 `web/media/` 并写入 `data/media.json`，无需手工准备。

## 自检

```bash
curl -s http://127.0.0.1:8849/api/samples | python3 -m json.tool   # 应列出 5 个示例
curl -s http://127.0.0.1:8848/api/samples | python3 -m json.tool   # 应列出值守视频
```

若 `samples` 为空，检查 `data/` 下 JSON 的 `file` 字段与 `web/assets/samples/` 是否对齐。
