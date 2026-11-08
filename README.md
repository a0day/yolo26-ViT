本仓库用于将vision transfomer集成进入yolo26 backbone，目前已有efficientViT EdgeViT。

---

**集成思路**：拉取yolo及目标vit仓库代码到本地/服务器，使用timm（如果timm里有对应模型）创建vit模型实例，创建 ViT -> YOLO26 的多尺度骨干适配器，修改 Ultralytics 模块注册和 parse_model，最后进行简单的前向、反向传播测试和一轮冒烟测试。根据不同的vit情况做额外处理。

---

edgevit.ipynb中有关于该流程的详细注释，可用于参考。