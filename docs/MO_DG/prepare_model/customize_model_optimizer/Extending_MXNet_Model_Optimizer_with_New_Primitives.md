# Extending Model Optimizer for Custom MXNet* Operations {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_MXNet_Model_Optimizer_with_New_Primitives}

This section provides instruction on how to support a custom MXNet operation (or as it called in the MXNet documentation
"operator" or "layer") which is not a part of the MXNet operation set. For example, if the operator is implemented using
the following [guide](https://mxnet.apache.org/versions/1.7.0/api/faq/new_op.html).

This section describes a procedure on how to extract operator attributes in the Model Optimizer. The rest of the
operation enabling pipeline and documentation on how to support MXNet operations from standard MXNet operation set is
described in the main document [Customize_Model_Optimizer](Customize_Model_Optimizer.md).

## Writing Extractor for Custom MXNet Operation
Custom MXNet operations have an attribute `op` (defining the type of the operation) equal to `Custom` and attribute
`op_type` which is an operation type defined by an user. Implement extractor class inherited from the
`MXNetCustomFrontExtractorOp` class instead of `FrontExtractorOp` class used for standard framework operations in order
to extract attributes for such kind of operations. The `op` class attribute value should be set to the `op_type` value
so the extractor is triggered for this kind of operation.

There is the example of the extractor for the custom operation registered with type (`op_type` value) equal to
`MyCustomOp` having attribute `my_attribute` of the floating point type with default value `5.6`. In this sample we
assume that we have already created the `CustomOp` class (inherited from `Op` class) for the Model Optimizer operation
for this MXNet custom operation as described in the [Customize_Model_Optimizer](Customize_Model_Optimizer.md).

```py
from extension.ops.custom_op import CustomOp  # implementation of the MO operation class
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import MXNetCustomFrontExtractorOp

class CustomProposalFrontExtractor(MXNetCustomFrontExtractorOp):  # inherit from specific base class
    op = 'MyCustomOp'  # the value corresponding to the `op_type` value of the MXNet operation
    enabled = True  # the extractor is enabled

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)  # parse the attributes to a dictionary with string values
        node_attrs = {
            'my_attribute': attrs.float('my_attribute', 5.6)
        }

        CustomOp.update_node_stat(node, node_attrs)  # update the attributes of the node
        return self.enabled
```
