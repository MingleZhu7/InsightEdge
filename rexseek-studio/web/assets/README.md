# 演示素材（不入库）

前端依赖以下素材，本仓库**不分发**（第三方图片/现场视频，版权与肖像需自行确认）。

## 通用演示（8849）

`web/assets/samples/` 下需要：

| 文件 | 对应示例 | 说明 |
|---|---|---|
| `demo9.jpg` | tomatoes 番茄成熟度 | 1083×1508 |
| `demo1.jpg` | proposal 桥上的求婚 | |
| `demo3.jpg` | webinar 讲者海报 | |
| `demo6.jpg` | team 团队合影 | |
| `helmet_all.jpg` / `helmet_mixed.jpg` | PPE 演示 | |

对应的元数据（标题、建议问法、预置框）已在 `data/samples_general.json` 里，**只缺图片文件**。

## 值守台（8848）

`web/media/` 下需要：

| 文件 | 说明 |
|---|---|
| `site_demo.mp4` / `site_demo.jpg` | 默认演示视频（1280×720，约 10 s）与其封面 |
| 其它 `site_*.mp4` | 可选，多个素材会在值守台左侧列出 |

第一次上传视频时，前端会把文件落到 `web/media/` 并写入 `data/media.json`，无需手工准备。

## 快速自检

```bash
curl -s http://127.0.0.1:8849/api/samples | python3 -m json.tool   # 应列出 5 个示例
curl -s http://127.0.0.1:8848/api/samples | python3 -m json.tool   # 应列出值守视频
```

若 `samples` 为空，检查 `data/samples_general.json` 里的 `file` 路径与 `web/assets/samples/` 是否对齐。
