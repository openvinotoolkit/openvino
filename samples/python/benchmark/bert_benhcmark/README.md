# Bert Benchmark Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_bert_benchmark_README}

This sample demonstrates how to estimate performace of a Bert model using Asynchronous Inference Request API. Unlike [demos](@ref omz_demos) this sample doesn't have configurable command line arguments. Feel free to modify sample's source code to try out different options.

The following Python\* API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| OpenVINO Runtime Version | [openvino.runtime.get_version] | Get Openvino API version |
| Basic Infer Flow | [openvino.runtime.Core], [openvino.runtime.Core.compile_model] | Common API to do inference: compile a model |
| Asynchronous Infer | [openvino.runtime.AsyncInferQueue], [openvino.runtime.AsyncInferQueue.start_async], [openvino.runtime.AsyncInferQueue.wait_all] | Do asynchronous inference |
| Model Operations | [openvino.runtime.CompiledModel.inputs] | Get inputs of a model |

## How It Works

The sample downloads a model and a tokenizer, export the model to onnx, reads the exported model and reshapes it to enforce dynamic inpus shapes, compiles the resulting model, downloads a dataset and runs benhcmarking on the dataset.

You can see the explicit description of
each sample step at [Integration Steps](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Running

Install the `openvino` Python package:

```
python -m pip install openvino
```

Install packages from `requirements.txt`:

```
python -m pip install -r requirements.txt
```

Run the sample

```
python bert_benhcmark.py
```

## Sample Output

The sample outputs how long it takes to process a dataset.

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
