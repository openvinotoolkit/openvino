# type: ignore
from __future__ import annotations
from openvino._pyopenvino import PartialShape
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.utils import refer_to_faq_msg
import openvino._pyopenvino
__all__ = ['Error', 'PartialShape', 'get_dimension_index_by_label', 'refer_to_faq_msg', 'update_layout_to_dict']
def get_dimension_index_by_label(input_shape: openvino._pyopenvino.PartialShape, input_names: list, layout_dict: [dict], dimension_label: str, default_dim: int):
    """
    
        The function returns index of the dimension pointed in the layout
        and a flag indicating if the index is chosen by default.
        For example, the index for 'D' dimension in "NHWDC" layout is 3.
        
    """
def update_layout_to_dict(inputs: list, layout: [list, dict], get_names_func: typing.Callable):
    """
    
        The function prepares layout values in the dictionary with items of the format:
        { node_name : {'source_layout': 'NHWC', 'target_layout': 'NCHW'} }
        
    """
