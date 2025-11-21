
# Tools

This file will be describing various auxiliary utilities designed to assist user in conducting post-inference analysis of artifacts produced by **multi-provider-inference-tool**

## multi_provider_blobs_comparison.py

This tool analyzes inference artefacts produced by different execution providers, listed as `ref_provider` and`providers_to_compare`, during an inference operation of a neuro `model` invocation.

The utils expects the following arguments:

 - `ref_provider`           - The name of a reference provider
 - `providers_to_compare`   - List of inference provider names, separated by a space symbol, which inference artifacts will be compared to the reference provider artifacts
 - `model`                  - A model name or a path to a file with a neuro model

#### Example:

    tools/multi_provider_blobs_comparison.py ov/CPU ov/NPU onnx/OpenVINOExecutionProvider/NPU resnet50

Will produce results of comparison of inference artifacts of these three providers `ov/CPU`, `ov/NPU`  and `onnx/OpenVINOExecutionProvider/NPU`, gathered on a previous step during their execution of the model `resnet50`.

<details>
<summary>Click to reveal a possible output</summary>

```
{
    "providers": [
        "ov/CPU",
        "ov/GPU"
    ],
    "data": {
        "ov/CPU": {
            "outputs": [
                "output-1-a"
            ],
            "data": {
                "output-1-a": {
                    "files": [
                        "odata_float32_[1_1000].blob"
                    ],
                    "shape": [
                        1,
                        1000
                    ],
                    "element_type": "float32",
                    "data": {
                        "odata_float32_[1_1000].blob": {
                            "path": "ov\\CPU\\0\\resnet-50-pytorch\\output\\495\\odata_float32_[1_1000].blob",
                            "std_correlation": 1
                        }
                    },
                    "not_found_data": {},
                    "status": []
                }
            },
            "not_found_data": {},
            "status": []
        },
        "ov/GPU": {
            "outputs": [
                "output-1-a"
            ],
            "data": {
                "output-1-a": {
                    "files": [
                        "odata_float32_[1_1000].blob"
                    ],
                    "shape": [
                        1,
                        1000
                    ],
                    "element_type": "float32",
                    "data": {
                        "odata_float32_[1_1000].blob": {
                            "path": "ov\\GPU\\0\\resnet-50-pytorch\\output\\495\\odata_float32_[1_1000].blob",
                            "std_correlation": 0.9882393240555067
                        }
                    },
                    "not_found_data": {},
                    "status": []
                }
            },
            "not_found_data": {},
            "status": []
        }
    },
    "status": [
        "Cannot find results for provider: ov/NPU, err: Cannot deserialize tensors as the provider directory doesn't exist: ov\\NPU\\0"
    ],
    "not_found_data": {
        "ov/NPU": {}
    }
}
```
</details>

### Result of post inference artifacts comparison:

Being executed, the utility makes multi-way comparison of model inference artifacts which consist of: meta information describing model outputs, meta information collecting output tensor attributes and blobs, which are representing tensors binary data.
This result resembles a tree and is represented by a JSON object described by the [multi provider result comparions schema](multi_provider_blobs_comparison_schema.json). It enumerates all providers, which are involved into the comparison, and describes model inference artifacts gathered for each provider from the requested comparison list. These artifacts include model outputs and output blobs and their properties accordingly. The comparison tree also stores list of inconsistent outputs and blobs which have no pairs among the artifacts produced by other providers.
Provided that output blobs have pairs among `ref_provider`'s post-inference artifacts, the following metrics will be added to the list of metrics of those output blobs.

### Metrics

#### NRMSE

It's the [Normalized Root Mean Square Error](https://en.wikipedia.org/wiki/Root_mean_square_deviation). Please follow the link to get more information.
Basically, it represents the similarity of data.
The value of the metric lies in the range `[0, 1]`
Where:
 - `0` - means that these two blobs contain completely different binary data
 - `1` - means that these two blobs has the same binary data, so that they are indistinguishable.

