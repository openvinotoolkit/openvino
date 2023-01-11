# OpenVINO™ 与 TensorFlow 集成 {#ovtf_integration_zh_CN}

**OpenVINO™ 与 TensorFlow 集成**是一个面向希望在其推理应用中开始使用 OpenVINO™ 的 TensorFlow 开发人员的解决方案。现在只需添加两行代码，即可在一系列英特尔® 计算设备上结合使用 OpenVINO™ 工具套件优化功能与 TensorFlow 推理应用。

这将满足您的全部需求：
```bash
import openvino_tensorflow
openvino_tensorflow.set_backend('<backend_name>')
```

**OpenVINO™ 与 TensorFlow 集成**可加快许多 AI 模型在采用各种英特尔® 技术时的推理速度，例如：
- 英特尔® CPU
- 英特尔® 集成 GPU
- 英特尔® Movidius™ 视觉处理器 - 称为 VPU
- 搭载 8 个英特尔® Movidius™ Myriad X 视觉处理器的英特尔® Vision Accelerator Design - 称为 VAD-M 或 HDDL

> **NOTE**: 为实现最佳性能、效率、工具自定义和硬件控制，建议开发人员采用原生 OpenVINO™ 解决方案。
> 如需查找有关该产品本身的更多信息，以及了解如何在项目中使用该产品，请访问其专用的 [GitHub 存储库](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/docs)。


如需了解通过 **OpenVINO™ 与 TensorFlow 集成**可执行哪些操作，请浏览 GitHub 存储库中[示例文件夹](https://github.com/openvinotoolkit/openvino_tensorflow/tree/master/examples)中的演示。

[英特尔® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/edge/build/ovtfoverview.html) 上还提供了示例教程。演示应用使用 Jupyter Notebook 实现。您可以在英特尔® DevCloud 节点上交互执行这些应用，比较 **OpenVINO™ 与 TensorFlow 集成**、原生 TensorFlow 和 OpenVINO™ 的结果。

## 许可
**OpenVINO™ 与 TensorFlow 集成**按照 [Apache 许可版本 2.0](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/LICENSE) 授予许可。
参与项目即表示您同意其中的许可和版权条款，并会根据这些条款进行参与。

## 支持

请通过 [GitHub 问题](https://github.com/openvinotoolkit/openvino_tensorflow/issues)提交您的问题、功能请求和错误报告。

## 如何参与

我们欢迎社区参与 **OpenVINO™ 与 TensorFlow 集成**。如有改进建议，请您：

* 通过 [GitHub 问题](https://github.com/openvinotoolkit/openvino_tensorflow/issues)分享您的提案。
* 提交[拉取请求](https://github.com/openvinotoolkit/openvino_tensorflow/pulls)。

我们会尽快审查您的参与情况。如果需要做出任何其他修复或修改，我们会为您提供指导和反馈。在您参与之前，请确保可以构建 **OpenVINO™ 与 TensorFlow 集成** 并运行所有包含修复/补丁的示例。如果希望推出重大功能，请为您的功能创建测试用例。在验证您的拉取请求后，我们会将其并入存储库，前提是该拉取请求符合上述要求并被证明可以接受。

---
\* 文中涉及的其它名称及商标属于各自所有者资产。