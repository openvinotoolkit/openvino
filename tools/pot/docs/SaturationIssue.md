# Saturation (overflow) Issue Workaround {#pot_saturation_issue}

## Introduction
8-bit instructions of previous generations of Intel&reg; CPUs, namely those based on SSE, AVX-2, AVX-512 instruction sets, admit so-called saturation (overflow) of the intermediate buffer when calculating the dot product which is an essential part of Convolutional or MatMul operations. This saturation can lead to an accuracy drop on the mentioned architectures during the inference of 8-bit quantized models. However, it is not possible to predict such degradation since most of the computations are executed in parallel during DL model inference which makes this process non-deterministic. This problem is typical for models with non-ReLU activation functions and low level of redundancy, for example, optimized or efficient models. It can prevent deploying the model on legacy hardware or creating cross-platform applications. The problem does not occur on the CPUs with Intel Deep Learning Boost (VNNI) technology and further generations, as well as on GPUs.

## Saturation Problem Detection
The only way to detect saturation issue is to run inference on the CPU that admits it and on the hardware that does not have such problem (for example, VNNI-based CPU). If the accuracy difference is significant (more than 1%), this is the main indicator of the saturation issue impact.

## Workaround
There is a workaround that helps fully address the saturation issue during the inference. The algorithm uses only 7 bits to represent weights (of Convolutional or Fully-Connected layers) while quantizing activations using the full range of 8-bit data types. However, this can lead to an accuracy degradation due to the reduced representation of weights. On the other hand, using this workaround for the first layer can help mitigate the saturation issue for many models.

POT tool provides three options to deal with the saturation issue. The options can be enabled in the POT configuration file using the "saturation_fix" parameter:

* (Default) Fix saturation issue for the first layer: "first_layer" option
* Apply for all layers in the model: "all" option
* Do not apply saturation fix at all: "no" option

Below is an example of the section in the POT configuration file with the `saturation_fix` option:
```json
"algorithms": [
    {
        "name": "DefaultQuantization",
        "params": {
            "preset": "performance",
            "stat_subset_size": 300,
            "saturation_fix": "all" // Apply the saturation fix to all the layers
        }
    }
]
```
## Recommendations

If you observe the saturation issue, we recommend trying the option "all" during the model quantization. If it does not help improve the accuracy, we recommend using [Quantization-aware training from NNCF](https://github.com/openvinotoolkit/nncf) and fine-tuning the model.

If you are not planning to use legacy CPU HW, you can use the option "no", which might also lead to slightly better accuracy.

## See Also
* [Lower Numerical Precision Deep Learning Inference and Training blogpost](https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html)
* [Configuration file description](@ref pot_configs_README)