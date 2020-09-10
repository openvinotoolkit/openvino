# Convert ONNX* DLRM to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_DLRM}

> **NOTE**: These instructions are currently deprecated. Since OpenVINO™ 2020.4 version, no specific steps are needed to convert ONNX\* DLRM models. For general instructions on converting ONNX models, please refer to [Converting a ONNX* Model](../Convert_Model_From_ONNX.md) topic.

These instructions are applicable only to the DLRM converted to the ONNX* file format from the [facebookresearch/dlrm model](https://github.com/facebookresearch/dlrm).

**Step 1**. Save trained Pytorch* model to ONNX* format. If you training model using [script provided in model repository](https://github.com/facebookresearch/dlrm/blob/master/dlrm_s_pytorch.py) just add `--save-onnx` flag to the command line parameters and you'll get `dlrm_s_pytorch.onnx` file containing model serialized in ONNX* format.

**Step 2**. To generate the Intermediate Representation (IR) of the model, change your current working directory to the Model Optimizer installation directory and run the Model Optimizer with the following parameters:
```sh
python3 ./mo.py --input_model dlrm_s_pytorch.onnx
```

Note that Pytorch model uses operation `torch.nn.EmbeddingBag`. This operation converts to onnx as custom `ATen` layer and not directly supported by OpenVINO*, but it is possible to convert this operation to:
* `Gather` if each "bag" consists of exactly one index. In this case `offsets` input becomes obsolete and not needed. They will be removed during conversion.
* `ExperimentalSparseWeightedSum` if "bags" contain not just one index. In this case Model Optimizer will print warning that pre-process of offsets is needed, because `ExperimentalSparseWeightedSum` and `torch.nn.EmbeddingBag` have different format of inputs.
For example if you have `indices` input of shape [indices_shape] and `offsets` input of shape [num_bags] you need to get offsets of shape [indices_shape, 2]. To do that you may use the following code snippet:
```python
import numpy as np

new_offsets = np.zeros((indices.shape[-1], 2), dtype=np.int32)
new_offsets[:, 1] = np.arange(indices.shape[-1])
bag_index = 0
for i in range(offsets.shape[-1] - 1):
    new_offsets[offsets[i]:offsets[i + 1], 0] = bag_index
    bag_index += 1
new_offsets[offsets[-1]:, 0] = bag_index
```
If you have more than one `torch.nn.EmbeddingBag` operation you'll need to do that for every offset input. If your offsets have same shape they will be merged into one input of shape [num_embedding_bags, indices_shape, 2].
