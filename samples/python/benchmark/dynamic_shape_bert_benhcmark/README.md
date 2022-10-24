# Dynamic shape Bert Benchmark Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_dynamic_shape_bert_benchmark_README}

This sample demonstrates how to estimate performace of a model using Asynchronous Inference Request API. Unlike [demos](@ref omz_demos) this sample doesn't have other configurable command line arguments. Feel free to modify sample's source code to try out different options.

The following Python\* API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| OpenVINO Runtime Version | [openvino.runtime.get_version] | Get Openvino API version |
| Basic Infer Flow | [openvino.runtime.Core], [openvino.runtime.Core.compile_model] | Common API to do inference: compile a model |
| Asynchronous Infer | [openvino.runtime.AsyncInferQueue], [openvino.runtime.AsyncInferQueue.start_async], [openvino.runtime.AsyncInferQueue.wait_all] | Do asynchronous inference |
| Model Operations | [openvino.runtime.CompiledModel.inputs] | Get inputs of a model |

| Options | Values |
| :--- | :--- |
| Validated Models | [bert-large-uncased-whole-word-masking-squad-0001](@ref omz_models_model_bert_large_uncased_whole_word_masking_squad_0001), [bert-large-uncased-whole-word-masking-squad-emb-0001](@ref omz_models_model_bert_large_uncased_whole_word_masking_squad_emb_0001) [bert-large-uncased-whole-word-masking-squad-int8-0001](@ref omz_models_model_bert_large_uncased_whole_word_masking_squad_int8_0001), [bert-small-uncased-whole-word-masking-squad-0001](@ref omz_models_model_bert_small_uncased_whole_word_masking_squad_0001), [bert-small-uncased-whole-word-masking-squad-0002](@ref omz_models_model_bert_small_uncased_whole_word_masking_squad_0002), [bert-small-uncased-whole-word-masking-squad-emb-int8-0001](@ref omz_models_model_bert_small_uncased_whole_word_masking_squad_emb_int8_0001), [bert-small-uncased-whole-word-masking-squad-int8-0002](@ref omz_models_model_bert_small_uncased_whole_word_masking_squad_int8_0002) |
| Model Format | OpenVINO™ toolkit Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx) |
| Supported devices | [All](../../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |

## How It Works

The sample reads a model ad reshapes it to enforce dynamic inpus shapes, compiles the resulting model, downloads a dataset and runs benhcmarking on the dataset.

You can see the explicit description of
each sample step at [Integration Steps](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md) section of "Integrate OpenVINO™ Runtime with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO™ Toolkit Samples guide.

## Running

```
python dynamic_shape_bert_benhcmark.py <path_to_model>
```

To run the sample, you need specify a model:

- you can use [public](@ref omz_models_group_public) or [Intel's](@ref omz_models_group_intel) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader).

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.

### Example

1. Install the `openvino-dev` Python package to use Open Model Zoo Tools:

```
python -m pip install openvino-dev[caffe,onnx,tensorflow2,pytorch,mxnet]
```

2. Download a pre-trained model using:

```
omz_downloader --name googlenet-v1
```

3. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:

```
omz_converter --name googlenet-v1
```

4. Perform benchmarking using the `googlenet-v1` model on a `CPU`:

```
python throughput_benchmark.py googlenet-v1.xml
```

## Sample Output

The application outputs how long it takes to process a dataset.

```
[ INFO ] OpenVINO:
         API version............. 2022.3.0-8261-37a0afa330f
Downloading builder script: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28.8k/28.8k [00:00<00:00, 14.4MB/s]
Downloading metadata: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28.7k/28.7k [00:00<00:00, 9.55MB/s]
Downloading readme: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22.0k/22.0k [00:00<00:00, 155kB/s]
Downloading and preparing dataset glue/sst2 (download: 7.09 MiB, generated: 4.81 MiB, post-processed: Unknown size, total: 11.90 MiB) to C:/Users/vzlobin/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.44M/7.44M [00:02<00:00, 3.38MB/s]
Dataset glue downloaded and prepared to C:/Users/vzlobin/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 229.13it/s]
[ INFO ] Duration:       78.07 seconds
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
