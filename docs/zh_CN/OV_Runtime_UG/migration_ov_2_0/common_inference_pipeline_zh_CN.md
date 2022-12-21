# 推理管道 {#openvino_2_0_inference_pipeline_zh_CN}

如需使用 OpenVINO™ 运行时推理模型，通常您需要在应用管道中执行以下步骤：
1. [创建核心对象](@ref create_core_zh_CN)。
   - 1.1.[（可选）加载扩展](@ref load_extensions_zh_CN)
2. [从驱动器中读取模型](@ref read_model_zh_CN)。
   - 2.1.[（可选）执行模型预处理](@ref perform_preprocessing_zh_CN)。
3. [将模型加载到设备](@ref load_model_to_device_zh_CN)。
4. [创建推理请求](@ref create_inference_request_zh_CN)。
5. [用数据填充输入张量](@ref fill_tensor_zh_CN)。
6. [开始推理](@ref start_inference_zh_CN)。
7. [处理推理结果](@ref process_results_zh_CN)。

以下代码将基于这些步骤演示如何更改应用代码以迁移到 API 2.0。

@anchor create_core_zh_CN
## 1. 创建核心对象

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_core
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_core
@endsphinxtab

@endsphinxtabset

@anchor load_extensions_zh_CN
### 1.1 （可选）加载扩展

如需通过自定义操作加载模型，您需要为这些操作添加扩展。强烈建议您使用 [OpenVINO™ 扩展性 API](@ref openvino_docs_Extensibility_UG_Intro_zh_CN) 编写扩展。但是，您也可以将旧扩展加载到新的 OpenVINO™ 运行时：

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:load_old_extension
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:load_old_extension
@endsphinxtab

@endsphinxtabset

@anchor read_model_zh_CN
## 2.从驱动器中读取模型

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:read_model
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:read_model
@endsphinxtab

@endsphinxtabset

读取模型采用与[模型创建迁移指南](@ref openvino_2_0_model_creation_zh_CN)中的示例相同的结构。

您可以在单次 `ov::Core::compile_model(filename, devicename)` 调用中组合模型读取和编译。

@anchor perform_preprocessing_zh_CN
### 2.1 （可选）执行模型预处理

当应用输入数据与模型输入格式不完全匹配时，可能需要进行预处理。请参阅 [API 2.0 中的预处理](@ref openvino_2_0_preprocessing_zh_CN)了解更多详情。

@anchor load_model_to_device_zh_CN
## 3.将模型加载到设备

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:compile_model
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:compile_model
@endsphinxtab

@endsphinxtabset

如果需要用 OpenVINO™ 运行时的其他参数配置设备，请参阅[配置设备](@ref openvino_2_0_configure_devices_zh_CN)。

@anchor create_inference_request_zh_CN
## 4.创建推理请求

**推理引擎 API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_infer_request
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_infer_request
@endsphinxtab

@endsphinxtabset

@anchor fill_tensor_zh_CN
## 5.用数据填充输入张量

**推理引擎 API**

推理引擎 API 用 `I32` 精度（与原始模型**不**一致）的数据填充输入：

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{使用代码创建的模型}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

API 2.0 用 `I64` 精度（与原始模型一致）的数据填充输入：

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{使用代码创建的模型}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

@anchor start_inference_zh_CN
## 6.开始推理

**推理引擎 API**

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:inference
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Async}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:start_async_and_wait
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:inference
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Async}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:start_async_and_wait
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

@anchor process_results_zh_CN
## 7.处理推理结果

**推理引擎 API**

推理引擎 API 处理输出的原因是，输出精度为 `I32`（与原始模型**不**一致）：

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{使用代码创建的模型}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

API 2.0 处理输出的原因如下，输出精度：

- 对于 OpenVINO IR v10 模型为 `I32`（与原始模型**不**一致），与[旧行为](@ref differences_api20_ie_zh_CN)匹配。
- 对于 OpenVINO IR v11、ONNX、ov::Model 和 PaddlePaddle 模型为 `I64`（与原始模型一致），与[新行为](@ref differences_api20_ie_zh_CN)匹配。


@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{使用代码创建的模型}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset
