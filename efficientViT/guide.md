# YOLO26 集成 EfficientViT 指南

适用环境：Google Colab T4、`/content/ultralytics`、Ultralytics 当前远程 `main`。

本方案采用：EfficientViT 输出 `[P3, P4, P5]` 列表，使用 Ultralytics 已有的 `Index` 模块取出三层特征。不要整段覆盖 `tasks.py`，也不需要强制改写 `_predict_once()`。

## 1. 新 Colab 运行时先拉取代码

在 `yolo.ipynb` 第一个代码单元执行。它会 clone：

```text
/content/ultralytics
/content/efficientvit
```

初始化单元使用远程 `main` 当前提交，不再依赖旧的 `1e4d3d2` 短哈希。执行完成后应检查：

```python
/content/ultralytics/ultralytics/cfg/models/26/yolo26.yaml
```

如果新的 GPU 运行时被删除，`/content` 会清空，需要重新执行初始化单元。

## 2. NumPy 和 YOLO 基线

初始化后执行基线单元，确认以下内容全部正常：

- NumPy 可以导入；
- `torch.cuda.is_available()` 为 `True`；
- GPU 显示为 T4；
- `YOLO("yolo26n.pt")` 可以加载。

如果出现：

```text
ImportError: cannot load module more than once per process
```

不要在同一个进程中 `reload(numpy)` 或重复安装 NumPy。保存 notebook 后选择 **运行时 → 断开连接并删除运行时**，重新连接后从第 1 步执行。

## 3. 安装并检查 timm

```python
%pip install -q timm

import timm
names = [x for x in timm.list_models() if "efficientvit_m" in x.lower()]
print(names)
assert "efficientvit_m0" in names
```

如果没有 `efficientvit_m0`，停止后再处理依赖版本。

## 4. 创建适配器

路径：

```text
/content/ultralytics/ultralytics/nn/modules/extramodules/efficientvit.py
```

适配器必须使用 `nn.Module` 类，而不是普通函数。核心结构如下：

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
        reductions = list(self.model.feature_info.reduction())
        channels = list(self.model.feature_info.channels())
        if reductions != [16, 32, 64]:
            raise RuntimeError(f"{variant} unexpected reductions: {reductions}")
        self.channel = [channels[0], channels[0], channels[1]]

    def forward(self, x):
        # timm EfficientViT-M 使用为 640x640 配置的固定窗口。
        # YOLO 构建阶段会用更小的临时输入计算 stride，因此先统一到 640，
        # 再把 P3/P4/P5 缩放回当前输入对应的尺寸。
        h, w = x.shape[-2:]
        x640 = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False)
        features = self.model(x640)
        p4 = features[0]
        p5 = features[1]
        p3 = F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = F.interpolate(p3, size=(max(h // 8, 1), max(w // 8, 1)), mode="nearest")
        p4 = F.interpolate(p4, size=(max(h // 16, 1), max(w // 16, 1)), mode="nearest")
        p5 = F.interpolate(p5, size=(max(h // 32, 1), max(w // 32, 1)), mode="nearest")
        return [p3, p4, p5]
```

M0 到 M5 应定义为类：

```python
class EfficientViT_M0(_EfficientViTBackbone):
    def __init__(self):
        super().__init__("efficientvit_m0", pretrained=False)
```

其他模型只需替换模型名。不能写成：

```python
def EfficientViT_M0():
    return _build("efficientvit_m0")
```

因为当前 `tasks.py` 会检查模块是否为 `torch.nn.Module` 子类。

## 5. 独立测试适配器

```python
backbone = EfficientViT_M0().cuda().eval()
x = torch.randn(1, 3, 640, 640, device="cuda")
with torch.inference_mode():
    outputs = backbone(x)

print(backbone.channel)
for output in outputs:
    print(tuple(output.shape))
```

预期空间尺寸：

```text
(80, 80)
(40, 40)
(20, 20)
```

## 6. 注册模块

在：

```text
/content/ultralytics/ultralytics/nn/modules/__init__.py
```

增加：

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

## 7. 修改 `tasks.py`

文件位置：

```text
/content/ultralytics/ultralytics/nn/tasks.py
```

### 7.1 增加导入

在文件顶部增加：

```python
from ultralytics.nn.modules.extramodules.efficientvit import (
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
)
```

### 7.2 增加模块集合

在 `repeat_modules` 后增加：

```python
efficientvit_modules = frozenset({
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
})
```

### 7.3 在解析循环中保留一次实例化

在下面这行后增加：

```python
n = n_ = max(round(n * depth), 1) if n > 1 else n
```

增加：

```python
m_ = None
```

将：

```python
if m in base_modules:
```

改为：

```python
if m in efficientvit_modules:
    if n != 1:
        raise ValueError("EfficientViT backbone only supports repeats=1")
    m_ = m(*args)
    c2 = m_.channel
elif m in base_modules:
```

将原来的：

```python
m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
```

改为：

```python
if m_ is None:
    m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
```

不要替换整个 `parse_model()`，也不要照旧指南强制改写 `_predict_once()`。

## 8. 创建 YAML

创建：

```text
/content/ultralytics/ultralytics/cfg/models/26/yolo26_efficientvit.yaml
```

第一版使用 M0：

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

`64, 64, 128` 必须与适配器打印的 `backbone.channel` 一致。

## 9. 重启并验证

完成第 6 至第 8 步后重启 Colab 运行时，再从初始化单元开始执行。然后运行：

```python
from ultralytics import YOLO

model = YOLO("/content/ultralytics/ultralytics/cfg/models/26/yolo26_efficientvit.yaml")
model.info(verbose=True)
```

再做前向测试：

```python
x = torch.randn(1, 3, 640, 640, device="cuda")
with torch.inference_mode():
    result = model(x)
print("forward passed")
```

常见错误：

| 错误 | 原因 |
|---|---|
| `module is not a permitted model layer` | EfficientViT 写成了函数，不是 `nn.Module` 类 |
| `KeyError: EfficientViT_M0` | `tasks.py` 没有导入模块 |
| `AttributeError: channel` | 适配器没有设置 `self.channel` |
| `Sizes of tensors must match` | P3/P4/P5 尺寸或 YAML 索引不一致 |
| `Expected input ... channels` | YAML 通道数与 `backbone.channel` 不一致 |

## 10. 训练前检查

```python
%cd /content/ultralytics
!git status --short
!git diff --stat
```

确认只改动预期文件，再进行 1 epoch 的 smoke test。`train.py` 暂不修改，除非出现明确的数据集或矩形验证错误。
