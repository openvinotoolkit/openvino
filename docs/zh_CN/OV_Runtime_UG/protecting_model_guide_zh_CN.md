# 结合使用加密模型和 OpenVINO™ {#openvino_docs_OV_UG_protecting_model_guide_zh_CN}

将深度学习功能部署到边缘设备可能会带来安全挑战，比如确保推理的完整性，或者为深度学习模型提供版权保护。

一种可能的解决方案是使用加密技术保护部署和存储在边缘设备上的模型。模型加密、解密和身份验证不是由 OpenVINO™ 提供的，但可以使用第三方工具（即 OpenSSL）实现。实现加密时，请确保使用最新版本的工具并遵循加密最佳实践。

本指南介绍如何通过受保护的模型安全地使用 OpenVINO。

## 安全部署模型

模型通过 OpenVINO™ 模型优化器优化后，会以 OpenVINO™ 中间表示 (OpenVINO™ IR) 格式部署到目标设备中。优化模型存储在边缘设备上，并通过 OpenVINO™ 运行时执行。
ONNX 和 PDPD 模型也可以通过 OpenVINO™ 运行时本机读取。

在将模型部署到边缘设备之前可对其进行加密和优化，以用于保护深度学习模型。边缘设备应始终保护存储的模型，并**仅在运行时**解密供 OpenVINO™ 运行时使用的模型。

![deploy_encrypted_model](../../OV_Runtime_UG/img/deploy_encrypted_model.svg)

## 加载加密模型

OpenVINO™ 运行时在加载之前需要进行模型解密。为模型解密分配一个临时内存块，并使用 `ov::Core::read_model` 方法从内存缓冲区加载模型。
有关更多信息，请参阅 `ov::Core` 类参考文档。

@snippet snippets/protecting_model_guide.cpp part0

英特尔® Software Guard Extensions（英特尔® SGX）等基于硬件的保护可用于保护解密操作机密，并将其绑定到设备上。有关更多信息，请参阅[英特尔® Software Guard Extensions](https://software.intel.com/en-us/sgx)。

使用 `ov::Core::read_model` 分别设置模型表示和权重。

目前，还没有办法从 ONNX 模型的内存中读取外部权重。
调用 `ov::Core::read_model(const std::string& model, const Tensor& weights)` 方法时，应将 `weights` 作为空值 `ov::Tensor` 传递。

@snippet snippets/protecting_model_guide.cpp part1

## 其他资源

- 英特尔® 发行版 OpenVINO™ 工具套件[主页](https://software.intel.com/en-us/openvino-toolkit)。
- 模型优化器[开发人员指南](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide_zh_CN.md)。
- [OpenVINO™ 运行时用户指南](openvino_intro_zh_CN.md)。
- 有关样本应用的更多信息，请参见 [OpenVINO™ 样本概述](../../../Samples_Overview.md)
- 如需获得有关一系列预训练模型的信息，请参见 [OpenVINO™ 工具套件预训练模型概述](@ref omz_models_group_intel)。
- 如需了解物联网库和代码样本，请参见[英特尔® 物联网开发套件](https://github.com/intel-iot-devkit)。
