# type: ignore
"""
openvino.experimental submodule
"""
from __future__ import annotations
import openvino._pyopenvino
__all__ = ['evaluate_as_partial_shape', 'evaluate_both_bounds', 'set_element_type', 'set_tensor_type']
def evaluate_as_partial_shape(output: openvino._pyopenvino.Output, partial_shape: openvino._pyopenvino.PartialShape) -> bool:
    """
                        Evaluates lower and upper value estimations for the output tensor. 
                        The estimation will be represented as a partial shape object, 
                        using Dimension(min, max) for each element.
    
                        :param output: Node output pointing to the tensor for estimation.
                        :type output: openvino.Output
                        :param partial_shape: The resulting estimation will be stored in this PartialShape.
                        :type partial_shape: openvino.PartialShape
                        :return: True if estimation evaluation was successful, false otherwise.
                        :rtype: bool
    """
def evaluate_both_bounds(output: openvino._pyopenvino.Output) -> tuple[openvino._pyopenvino.Tensor, openvino._pyopenvino.Tensor]:
    """
                        Evaluates lower and upper value estimations of the output tensor.
                        It traverses the graph upwards to deduce the estimation.
    
                        :param output: Node output pointing to the tensor for estimation.
                        :type output: openvino.Output
                        :return: Tensors representing the lower and upper bound value estimations.
                        :rtype: Tuple[openvino.Tensor, openvino.Tensor]
    """
def set_element_type(tensor: openvino._pyopenvino.DescriptorTensor, element_type: openvino._pyopenvino.Type) -> None:
    """
                        Sets element type for a tensor descriptor in the OV model graph.
    
                        :param tensor: The tensor descriptor whose element type is to be set.
                        :type tensor: openvino.Tensor 
                        :param element_type: A new element type of the tensor descriptor.
                        :type element_type: openvino.Type
    """
def set_tensor_type(tensor: openvino._pyopenvino.DescriptorTensor, element_type: openvino._pyopenvino.Type, partial_shape: openvino._pyopenvino.PartialShape) -> None:
    """
                        Changes element type and partial shape of a tensor descriptor in the OV model graph.
    
                        :param tensor: The tensor descriptor whose element type is to be set.
                        :type tensor: openvino.Tensor 
                        :param element_type: A new element type of the tensor descriptor.
                        :type element_type: openvino.Type
                        :param partial_shape: A new partial shape of the tensor desriptor.
                        :type partial_shape: openvino.PartialShape
    """
