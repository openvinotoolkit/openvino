# Extending the MXNet Model Optimizer with New Primitives  {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_MXNet_Model_Optimizer_with_New_Primitives}

This section describes how you can create a Model Optimizer extension for a custom layer from your MXNet* model. It supplements the main document [Extending Model Optimizer with New Primitives](Extending_Model_Optimizer_with_New_Primitives.md) and provides a step-by-step procedure. To create an extension for a particular layer, perform the following steps:

1.  Create the file `custom_proposal_ext.py` in the folder `<INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/mxnet`
If your MXNet layer has op `Custom`, create the `CustomProposalFrontExtractor` class inherited from `MXNetCustomFrontExtractorOp`:
```py
from mo.front.extractor import MXNetCustomFrontExtractorOp
class CustomProposalFrontExtractor(MXNetCustomFrontExtractorOp):
    pass
```
Otherwise, for layers that are not standard MXNet layers, create the `ProposalFrontExtractor` class inherited from `FrontExtractorOp`:
```py
    from mo.front.extractor import FrontExtractorOp
    class ProposalFrontExtractor(FrontExtractorOp):
        pass
```
2.  Specify the operation that the extractor refers to and a specific flag. The flag represents whether the operation should be used by the Model Optimizer or should be excluded from processing:
```py
from mo.front.extractor import MXNetCustomFrontExtractorOp
class CustomProposalFrontExtractor(MXNetCustomFrontExtractorOp):
    op = '_contrib_Proposal'
    enabled = True
```
3.  Register a mapping rule between the original model and the `PythonProposalOp` attributes by overriding the following function:
```py
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import MXNetCustomFrontExtractorOp
from mo.ops.op import Op

class CustomProposalFrontExtractor(MXNetCustomFrontExtractorOp):
    op = '_contrib_Proposal'
    enabled = True
    @staticmethod
    def extract(node):
    attrs = get_mxnet_layer_attrs(node.symbol_dict)
        node_attrs = {
            'feat_stride': attrs.float('feat_stride', 16)
        }
        
        # update the attributes of the node
        Op.get_op_class_by_name('Proposal').update_node_stat(node, node_attrs) # <------ here goes the name ('Proposal') of the Operation that was implemented before
        return __class__.enabled
```

