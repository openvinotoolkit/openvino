# Filter Pruning of Convolutional Models {#filter_pruning}

## Introduction
Filter pruning is an advanced optimization method which allows reducing computational complexity of the model by removing redundant or unimportant filters from convolutional operations of the model. This removal is done in two steps: 
1. Unimportant filters are zeroed out by the NNCF optimization with fine-tuning.
2. Zero filters are removed from the model during the export to OpenVINO&trade; Intermediate Representation (IR).

Filter Pruning method from the NNCF can be used stand-alone but we usually recommend to stack it with 8-bit quantization for two reasons. First, 8-bit quantization is the best method in terms of achieving the highest accuracy-performance trade-offs so stacking it with filter pruning can give even better performance results. Second, applying quantization along with filter pruning does not hurt accuracy a lot since filter pruning removes noisy filters from the model which narrows down values ranges of weights and activations and helps to reduce overall quantization error.

> **NOTE**: Filter Pruning usually requires a long fine-tuning or retraining of the model which can be comparable to training the model from scratch. Otherwise, a large accuracy degradation can be caused. Therefore, the training schedule should be adjusted accordingly when applying this method. 

Below, we provide the steps that are required to apply Filter Pruning + QAT to the model:

## Applying Filter Pruning with fine-tuning
Here, we show the basic steps to modify the training script for the model and use it to zero out unimportant filters:

### 1. Import NNCF API
In this step, NNCF-related imports are added in the beginning of the training script:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/pruning_torch.py imports

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/pruning_tf.py imports

@endsphinxtab

@endsphinxtabset

### 2. Create NNCF configuration
Here, you should define NNCF configuration which consists of model-related parameters (`"input_info"` section) and parameters of optimization methods (`"compression"` section). 

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/pruning_torch.py nncf_congig

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/pruning_tf.py nncf_congig

@endsphinxtab

@endsphinxtabset

Here is a brief description of the required parameters of the Filter Pruning method. For full description refer to the [GitHub](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Pruning.md) page.
- `pruning_init` - initial pruning rate target. For example, value `0.1` means that at the begging of training, convolutions that can be pruned will have 10% of their filters set to zero.
- `pruning_target` - pruning rate target at the end of the schedule. For example, the value `0.5` means that at the epoch with the number of `num_init_steps + pruning_steps`, convolutions that can be pruned will have 50% of their filters set to zero.
- `pruning_steps` - the number of epochs during which the pruning rate target is increased from `pruning_init` to `pruning_target` value. We recommend to keep the highest learning rate during this period.

### 3. Apply optimization methods
In the next step, the original model is wrapped by the NNCF object using the `create_compressed_model()` API using the configuration defined in the previous step. This method returns a so-called compression controller and the wrapped model that can be used the same way as the original model. It is worth noting that optimization methods are applied at this step so that the model undergoes a set of corresponding transformations and can contain additional operations required for the optimization. 
@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/pruning_torch.py wrap_model

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/pruning_tf.py wrap_model

@endsphinxtab

@endsphinxtabset

### 4. Fine-tune the model
This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle, you can skip this step which means that the post-training optimization will be applied to the model.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/pruning_torch.py tune_model

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/pruning_tf.py tune_model

@endsphinxtab

@endsphinxtabset



## Removing zero filters from the pruned model