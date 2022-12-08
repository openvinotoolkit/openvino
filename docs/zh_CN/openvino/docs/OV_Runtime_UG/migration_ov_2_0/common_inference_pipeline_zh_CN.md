# 推理管道{#openvino_2_0_inference_pipeline_zh_CN}

如需使用 OpenVINO™ 运行时推理模型，通常您需要在应用管道中执行以下步骤：
1. <a href="#create-core">创建核心对象。</a>
   - 1.1. <a href="#load-extensions">（可选）加载扩展。</a>
2. <a href="#read-model">从驱动器中读取模型。</a>
   - 2.1. <a href="#perform-preprocessing">（可选）执行模型预处理。</a>
3. <a href="#load-model-to-device">将模型加载到设备。</a>
4. <a href="#create-inference-request">创建推理请求。</a>
5. <a href="#fill-tensor">用数据填充输入张量。</a>
6. <a href="#start-inference">开始推理。</a>
7. <a href="#process-results">处理推理结果。</a>

以下代码将基于这些步骤演示如何更改应用代码以迁移到 API 2.0。

## <a name="create-core"></a>1. 创建核心对象

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

### <a name="load-extensions"></a>1.1 （可选）加载扩展

如需通过自定义操作加载模型，您需要为这些操作添加扩展。强烈建议您使用 [OpenVINO™ 扩展性 API](../../Extensibility_UG/Intro_zh_CN.md) 编写扩展。但是，您也可以将旧扩展加载到新的 OpenVINO™ 运行时：

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

## <a name="read-model"></a>2.从驱动器中读取模型

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

读取模型采用与[模型创建迁移指南](./graph_construction_zh_CN.md)中的示例相同的结构。

您可以在单次 `ov::Core::compile_model(filename, devicename)` 调用中组合模型读取和编译。

### <a name="perform-preprocessing"></a>2.1 （可选）执行模型预处理

当应用输入数据与模型输入格式不完全匹配时，可能需要进行预处理。请参阅 [API 2.0 中的预处理](./preprocessing_zh_CN.md)了解更多详情。

## <a name="load-model-to-device"></a>3.将模型加载到设备

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

如果需要用 OpenVINO™ 运行时的其他参数配置设备，请参阅[配置设备](./configure_devices_zh_CN.md)。

## <a name="create-inference-request"></a>4.创建推理请求

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

## <a name="fill-tensor"></a>5.用数据填充输入张量

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

## <a name="start-inference"></a>6.开始推理

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

## <a name="process-results"></a>7.处理推理结果

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

API 2.0 处理输出的原因如下：

- 对于 OpenVINO™ IR v10 模型，输出精度为 `I32`（与原始模型**不**一致），目的是匹配 <a href="openvino_2_0_transition_guide#differences-api20-ie">旧行为</a>。
- 对于 OpenVINO™ IR v11、ONNX、ov::Model 和 PaddlePaddle 模型，由于输出精度为 `I64`（与原始模型一致），目的是匹配 <a href="openvino_2_0_transition_guide#differences-api20-ie">新行为</a>。

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
