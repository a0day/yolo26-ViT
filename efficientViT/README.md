# YOLO26 + EfficientViT

本项目记录在 Google Colab T4 上，将 EfficientViT-M0 集成到 Ultralytics YOLO26 的流程。

## 当前实现

- YOLO26 来源：`/content/ultralytics`
- EfficientViT 原始仓库：`/content/efficientvit`
- 实际接入方式：通过 `timm` 创建 `efficientvit_m0`
- 适配器输出：`[P3, P4, P5]`
- YOLO 配置：`ultralytics/cfg/models/26/yolo26_efficientvit.yaml`
- 当前验证：模型构建和 640×640 前向推理已通过

原始 `/content/efficientvit` 仓库保留为独立目录，不直接复制到 Ultralytics 内部，也不覆盖 YOLO26 原始配置。

## 一、Colab 初始化

在新的 Colab T4 运行时中执行：

```python
from pathlib import Path
import subprocess

CONTENT = Path("/content")
ULTRALYTICS = CONTENT / "ultralytics"
EFFICIENTVIT = CONTENT / "efficientvit"

def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)

if not (ULTRALYTICS / ".git").exists():
    run("git clone --depth 1 https://github.com/ultralytics/ultralytics.git /content/ultralytics")
if not (EFFICIENTVIT / ".git").exists():
    run("git clone --depth 1 https://github.com/mit-han-lab/efficientvit.git /content/efficientvit")

run("git fetch --depth 1 origin main", cwd=ULTRALYTICS)
run("git checkout --detach origin/main", cwd=ULTRALYTICS)
run("python -m pip install -q -e /content/ultralytics --no-deps")
```

确认 YOLO26 配置存在：

```python
assert Path("/content/ultralytics/ultralytics/cfg/models/26/yolo26.yaml").exists()
```

如果 Colab 运行时被删除，`/content` 会清空，需要重新执行初始化单元。

## 二、NumPy 和 YOLO 基线

```python
%cd /content/ultralytics

import sys
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Ultralytics:", ultralytics.__file__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

baseline = YOLO("yolo26n.pt")
print("baseline model loaded")
```

如果出现 `cannot load module more than once per process`，保存 notebook 后选择 **运行时 → 断开连接并删除运行时**，重新连接，再从初始化单元开始执行。不要在同一个 Python 进程中重复安装 NumPy 或 `reload(numpy)`。

## 三、安装并检查 timm

```python
%pip install -q timm

import timm
names = [name for name in timm.list_models() if "efficientvit_m" in name.lower()]
print(names)
assert "efficientvit_m0" in names
```

## 四、创建 EfficientViT 适配器

文件位置：

```text
/content/ultralytics/ultralytics/nn/modules/extramodules/efficientvit.py
```

适配器完成以下工作：

1. 使用 `timm.create_model(..., features_only=True)` 创建 EfficientViT-M；
2. 固定内部输入为 640×640，适配 EfficientViT-M 的窗口限制；
3. 取得模型的中间特征；
4. 将特征调整为 YOLO 所需的 P3、P4、P5 尺寸；
5. 暴露 `self.channel`，供 YOLO 模型解析器读取通道数。

核心结构：

```python
class _EfficientViTBackbone(nn.Module):
    def __init__(self, variant, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            img_size=640,
            features_only=True,
            out_indices=(0, 1, 2),
        )
        self.channel = [channels[0], channels[0], channels[1]]

    def forward(self, x):
        h, w = x.shape[-2:]
        x640 = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False)
        features = self.model(x640)
        # 返回 [P3, P4, P5]
        return [p3, p4, p5]
```

M0 到 M5 定义为 `nn.Module` 子类，例如：

```python
class EfficientViT_M0(_EfficientViTBackbone):
    def __init__(self):
        super().__init__("efficientvit_m0", pretrained=False)
```

不能把 `EfficientViT_M0` 写成普通工厂函数，因为当前 YOLO 模型解析器会检查模块类型。

## 五、独立测试适配器

```python
from ultralytics.nn.modules.extramodules.efficientvit import EfficientViT_M0

backbone = EfficientViT_M0().cuda().eval()
x = torch.randn(1, 3, 640, 640, device="cuda")

with torch.inference_mode():
    outputs = backbone(x)

print("channels:", backbone.channel)
for output in outputs:
    print(tuple(output.shape))
```

预期空间尺寸为：

```text
P3: 80 × 80
P4: 40 × 40
P5: 20 × 20
```

## 六、注册模块并修改模型解析器

在 `ultralytics/nn/modules/__init__.py` 增加：

```python
from .extramodules.efficientvit import (
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
)
```

在 `ultralytics/nn/tasks.py` 中：

1. 导入六个 EfficientViT 类；
2. 建立 `efficientvit_modules` 集合；
3. 在 `parse_model()` 中单独处理 EfficientViT；
4. 实例化一次，读取 `m_.channel`；
5. 保留当前版本的其他解析逻辑；
6. 不要整段替换 `parse_model()`；
7. 不需要强制改写 `_predict_once()`。

EfficientViT 分支的核心逻辑是：

```python
if m in efficientvit_modules:
    if n != 1:
        raise ValueError("EfficientViT backbone only supports repeats=1")
    m_ = m(*args)
    c2 = m_.channel
elif m in base_modules:
    ...
```

普通模块实例化保留为：

```python
if m_ is None:
    m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
```

修改前的 `__init__.py` 和 `tasks.py` 可分别保存为 `.bak` 文件，便于回退。

## 七、创建 YOLO26 配置

文件：

```text
ultralytics/cfg/models/26/yolo26_efficientvit.yaml
```

当前 M0 配置为：

```yaml
nc: 80
end2end: True
reg_max: 1

backbone:
  - [-1, 1, EfficientViT_M0, []]

head:
  - [0, 1, Index, [64, 0]]
  - [0, 1, Index, [64, 1]]
  - [0, 1, Index, [128, 2]]
  - [[1, 2, 3], 1, Detect, [nc]]
```

`64, 64, 128` 必须与适配器实际打印的 `backbone.channel` 一致。

## 八、重启运行时并构建模型

完成 Python 文件修改后重启 Colab 运行时，清除旧的导入缓存。重启后重新执行初始化和依赖单元，再运行：

```python
from ultralytics import YOLO

model = YOLO(
    "/content/ultralytics/ultralytics/cfg/models/26/yolo26_efficientvit.yaml"
)
model.info(verbose=True)
```

当前 notebook 的实际结果：

```text
YOLO26_efficientvit summary: 329 layers, 2,999,832 parameters
```

## 九、前向推理验证

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
x = torch.rand(1, 3, 640, 640, device=device)

with torch.inference_mode():
    result = model(x)

print("forward passed")
```

使用 `torch.rand()` 生成 0 到 1 的输入，避免原始随机张量触发输入归一化警告。

## 十、训练前和 smoke test

先检查改动：

```python
%cd /content/ultralytics
!git status --short
!git diff --stat
```

然后进行 1 个 epoch 的小规模训练：

```python
model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=640,
    batch=2,
    workers=2,
    device=0,
    project="/content/runs",
    name="yolo26_efficientvit_smoke",
)
```

当前方案暂不修改 `train.py`。只有出现明确的数据集或矩形验证错误时，才单独处理 `build_dataset()`。

## 十一、已知限制和后续工作

- 当前配置使用 `EfficientViT_M0`，M1 到 M5 只完成了适配器接口定义；
- 当前适配器使用 `pretrained=False`，正式实验需要明确初始化策略；
- P3 是从 EfficientViT 中间特征上采样得到的，需要通过实验验证其检测效果；
- 需要完成 smoke test、checkpoint 重载、正式训练和验证集评估；
- 应与原始 YOLO26n 比较 mAP、参数量、显存和推理速度；
- 正式训练前应记录使用的 Ultralytics、PyTorch、timm、CUDA 和 GPU 版本。

## 十二、相关文件

```text
yolo.ipynb                         # Colab 执行流程
guide.md                          # 分步骤操作指南
ultralytics/nn/tasks.py           # EfficientViT 模块解析逻辑
ultralytics/nn/modules/__init__.py
ultralytics/nn/modules/extramodules/efficientvit.py
ultralytics/cfg/models/26/yolo26_efficientvit.yaml
```
