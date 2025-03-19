# type: ignore
"""
Factory functions for all openvino ops.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from openvino.utils.types import make_constant_node
import functools
import numpy as np
import openvino._pyopenvino
import typing
__all__ = ['Node', 'NodeInput', 'as_node', 'as_nodes', 'eye', 'generate_proposals', 'grid_sample', 'irdft', 'make_constant_node', 'multiclass_nms', 'nameable_op', 'non_max_suppression', 'np', 'partial', 'rdft', 'roi_align', 'softsign']
def eye(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs eye operation.
    
        :param num_rows: The node providing row number tensor.
        :param num_columns: The node providing column number tensor.
        :param diagonal_index: The node providing the index of the diagonal to be populated.
        :param output_type: Specifies the output tensor type, supports any numeric types.
        :param batch_shape: The node providing the leading batch dimensions of output shape. Optionally.
        :param name: The optional new name for output node.
        :return: New node performing deformable convolution operation.
        
    """
def generate_proposals(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs GenerateProposals operation.
    
        :param im_info: Input with image info.
        :param anchors: Input anchors.
        :param deltas: Input deltas.
        :param scores: Input scores.
        :param min_size: Specifies minimum box width and height.
        :param nms_threshold: Specifies threshold to be used in the NMS stage.
        :param pre_nms_count: Specifies number of top-n proposals before NMS.
        :param post_nms_count: Specifies number of top-n proposals after NMS.
        :param normalized: Specifies whether proposal bboxes are normalized or not. Optional attribute, default value is `True`.
        :param nms_eta: Specifies eta parameter for adaptive NMS., must be in range `[0.0, 1.0]`. Optional attribute, default value is `1.0`.
        :param roi_num_type: Specifies the element type of the third output `rpnroisnum`. Optional attribute, range of values: `i64` (default) or `i32`.
        :param name: The optional name for the output node.
        :return: New node performing GenerateProposals operation.
        
    """
def grid_sample(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], grid: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], attributes: dict, name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Return a node which performs GridSample operation.
    
        :param data: The input image.
        :param grid: Grid values (normalized input coordinates).
        :param attributes: A dictionary containing GridSample's attributes.
        :param name: Optional name of the node.
    
        Available attributes:
    
        * align_corners A flag which specifies whether to align the grid extrema values
                        with the borders or center points of the input tensor's border pixels.
                        Range of values: true, false
                        Default value: false
                        Required: no
        * mode          Specifies the type of interpolation.
                        Range of values: bilinear, bicubic, nearest
                        Default value: bilinear
                        Required: no
        * padding_mode  Specifies how the out-of-bounds coordinates should be handled.
                        Range of values: zeros, border, reflection
                        Default value: zeros
                        Required: no
    
        :return: A new GridSample node.
        
    """
def irdft(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs IRDFT operation.
    
        :param data: Tensor with data.
        :param axes: Tensor with axes to transform.
        :param signal_size: Optional tensor specifying signal size with respect to axes from the input 'axes'.
        :param name: Optional output node name.
        :return: The new node which performs IRDFT operation on the input data tensor.
        
    """
def multiclass_nms(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs MulticlassNms.
    
        :param boxes: Tensor with box coordinates.
        :param scores: Tensor with box scores.
        :param roisnum: Tensor with roisnum. Specifies the number of rois in each image. Required when
                        'scores' is a 2-dimensional tensor.
        :param sort_result_type: Specifies order of output elements, possible values:
                                 'class': sort selected boxes by class id (ascending)
                                 'score': sort selected boxes by score (descending)
                                 'none': do not guarantee the order.
        :param sort_result_across_batch: Specifies whenever it is necessary to sort selected boxes
                                         across batches or not
        :param output_type: Specifies the output tensor type, possible values:
                            'i64', 'i32'
        :param iou_threshold: Specifies intersection over union threshold
        :param score_threshold: Specifies minimum score to consider box for the processing
        :param nms_top_k: Specifies maximum number of boxes to be selected per class, -1 meaning
                          to keep all boxes
        :param keep_top_k: Specifies maximum number of boxes to be selected per batch element, -1
                           meaning to keep all boxes
        :param background_class: Specifies the background class id, -1 meaning to keep all classes
        :param nms_eta: Specifies eta parameter for adpative NMS, in close range [0, 1.0]
        :param normalized: Specifies whether boxes are normalized or not
        :param name: The optional name for the output node
        :return: The new node which performs MuticlassNms
        
    """
def non_max_suppression(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs NonMaxSuppression.
    
        :param boxes: Tensor with box coordinates.
        :param scores: Tensor with box scores.
        :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                            to be selected per class.
        :param iou_threshold: Tensor specifying intersection over union threshold
        :param score_threshold: Tensor specifying minimum score to consider box for the processing.
        :param soft_nms_sigma: Tensor specifying the sigma parameter for Soft-NMS.
        :param box_encoding: Format of boxes data encoding.
        :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                       boxes across batches or not.
        :param output_type: Output element type.
        :return: The new node which performs NonMaxSuppression
        
    """
def rdft(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs RDFT operation.
    
        :param data: Tensor with data.
        :param axes: Tensor with axes to transform.
        :param signal_size: Optional tensor specifying signal size with respect to axes from the input 'axes'.
        :param name: Optional output node name.
        :return: The new node which performs RDFT operation on the input data tensor.
        
    """
def roi_align(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], rois: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], batch_indices: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], pooled_h: int, pooled_w: int, sampling_ratio: int, spatial_scale: float, mode: str, aligned_mode: typing.Optional[str] = 'asymmetric', name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Return a node which performs ROIAlign operation.
    
        :param data: Input data.
        :param rois: RoIs (Regions of Interest) to pool over.
        :param batch_indices: Tensor with each element denoting the index of
                              the corresponding image in the batch.
        :param pooled_h: Height of the ROI output feature map.
        :param pooled_w: Width of the ROI output feature map.
        :param sampling_ratio: Number of bins over height and width to use to calculate
                               each output feature map element.
        :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
        :param mode: Method to perform pooling to produce output feature map elements. Avaiable modes are:
                             - 'max' - maximum pooling
                             - 'avg' - average pooling
        :param aligned_mode: Specifies how to transform the coordinate in original tensor to the resized tensor.
                             Mode 'asymmetric' is the default value. Optional. Avaiable aligned modes are:
                             - 'asymmetric'
                             - 'half_pixel_for_nn'
                             - 'half_pixel'
        :param name: The optional name for the output node
    
        :return: The new node which performs ROIAlign
        
    """
def softsign(node: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Apply SoftSign operation on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: The optional name for the output node.
        :return: New node with SoftSign operation applied on each element of it.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset9: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset9')
