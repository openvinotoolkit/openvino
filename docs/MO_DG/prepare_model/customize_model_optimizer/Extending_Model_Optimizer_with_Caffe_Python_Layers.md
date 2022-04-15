# Extending Model Optimizer with Caffe* Python Layers {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_Model_Optimizer_With_Caffe_Python_Layers}

This section provides instructions on how to support a custom, written only in Python, Caffe operations, e.g., the
[Faster-R-CNN model](http://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0) implemented in
Caffe contains a custom proposal layer written in Python. That layer is described in the
[Faster-R-CNN prototxt](https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt) as:
```sh
layer {
  name: 'proposal'
  type: 'Python'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rois'
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16"
  }
}
```

Below section describes only a procedure on how to extract operator attributes in the Model Optimizer. The rest of the enabling pipeline operations and documentation on how to support other (written in C++) Caffe operations are described in
the main [Customize_Model_Optimizer](Customize_Model_Optimizer.md) article.

## Writing Extractor for Caffe Python Layer
Custom Caffe Python layers have an attribute `type` (defining the type of the operation) equal to `Python` and two
mandatory attributes in the `python_param` dictionary: `module` and `layer`. The `module` defines the Python module name
with the layer implementation, while `layer` value is an operation type defined by an user. To extract
attributes for such operation, it is necessary to implement extractor class inherited from the
`CaffePythonFrontExtractorOp` class instead of `FrontExtractorOp` class used for standard framework layers. The `op`
class attribute value should be set to the `module + "." + layer` value so the extractor is triggered for this kind of
operation.

Below is a simplified example of the extractor for the custom operation Proposal from previously mentioned Faster-R-CNN model.
The full code with additional checks is provided in [this](https://github.com/openvinotoolkit/openvino/blob/releases/2022/1/tools/mo/openvino/tools/mo/front/caffe/proposal_python_ext.py) file. The sample code uses
operation `ProposalOp`, which corresponds to `Proposal` operation described in the [Available Operations Sets](../../../ops/opset.md)
article. Refer to the source code below for a detailed explanation of the extractor.

```py
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.extractor import CaffePythonFrontExtractorOp


class ProposalPythonFrontExtractor(CaffePythonFrontExtractorOp):
    op = 'rpn.proposal_layer.ProposalLayer'  # module + "." + layer
    enabled = True  # extractor is enabled

    @staticmethod
    def extract_proposal_params(node, defaults):
        param = node.pb.python_param  # get the protobuf message representation of the layer attributes
        # parse attributes from the layer protobuf message to a Python dictionary
        attrs = CaffePythonFrontExtractorOp.parse_param_str(param.param_str)
        update_attrs = defaults

        # the operation expects ratio and scale values to be called "ratio" and "scale" while Caffe uses different names
        if 'ratios' in attrs:
            attrs['ratio'] = attrs['ratios']
            del attrs['ratios']
        if 'scales' in attrs:
            attrs['scale'] = attrs['scales']
            del attrs['scales']

        update_attrs.update(attrs)
        ProposalOp.update_node_stat(node, update_attrs)  # update the node attributes

    @classmethod
    def extract(cls, node):
        # define default values for the Proposal layer attributes
        defaults = {
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7
        }
        cls.extract_proposal_params(node, defaults)
        return cls.enabled
```

## See Also

* [Customize_Model_Optimizer](Customize_Model_Optimizer.md)
