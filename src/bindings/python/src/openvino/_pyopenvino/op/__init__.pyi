# type: ignore
from . import util
from __future__ import annotations
import collections.abc
import numpy
import openvino._pyopenvino
import typing
"""
Package ngraph.impl.op that wraps ov::op
"""
__all__ = ['Constant', 'Parameter', 'Result', 'assign', 'if_op', 'loop', 'read_value', 'tensor_iterator', 'util']
class Constant(openvino._pyopenvino.Node):
    """
    openvino.op.Constant wraps ov::op::v0::Constant
    """
    @typing.overload
    def __init__(self, array: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], shared_memory: bool = False) -> None:
        ...
    @typing.overload
    def __init__(self, tensor: openvino._pyopenvino.Tensor, shared_memory: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[str]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[...]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape, arg2: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_byte_size(self) -> int:
        ...
    def get_data(self, *, dtype: typing.Any = None, copy: bool = False) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        """
                    Access to Constant's data. Returns numpy array with corresponding shape.
        
                    Function tries to return a view by default, if not possible due
                    to types mismatch (between the Constant's type and `dtype`)
                    or when `copy=True`, then make a copy of data.
        
                    If `dtype` is not specified, it's inherited from Constant itself.
        
                    For Constants with OpenVINO specific element type, such as u1,
                    it returns linear array (as view) with uint8 / int8 numpy dtype.
                    In such cases if `dtype` is used, function also creates a copy and
                    unpacks the data.
        
                    Note: can be used to upcast BF16 data type to float32 or float64. 
        
                    :param dtype: Targeted data type.
                    :type dtype: numpy.dtype, optional, keyword-only
                    :param copy: Enable or disable copy of data.
                    :type copy: bool, optional, keyword-only
                    :rtype: numpy.array
        """
    def get_strides(self) -> openvino._pyopenvino.Strides:
        """
                            Constant's strides in bytes.
        
                            :rtype: openvino.Strides
        """
    def get_tensor_view(self) -> openvino._pyopenvino.Tensor:
        """
                            Get view on constant data as tensor.
        
                            :rtype: openvino.Tensor
        """
    def get_value_strings(self) -> list[str]:
        ...
    def get_vector(self) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        ...
    @property
    def data(self) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]:
        """
                    Access to Constant's data - creates a view of data.
        
                    Returns numpy array with corresponding shape and dtype.
                    For Constants with openvino specific element type, such as u1,
                    it returns linear array, with uint8 / int8 numpy dtype.
        
                    Note: this access method reflects shared memory if it was applied during initialization.
        
                    :rtype: numpy.array
        """
    @property
    def strides(self) -> openvino._pyopenvino.Strides:
        """
                                            Constant's strides in bytes.
        
                                            :rtype: openvino.Strides
        """
    @property
    def tensor_view(self) -> openvino._pyopenvino.Tensor:
        """
                                            Get view on constant data as tensor.
        
                                            :rtype: openvino.Tensor
        """
class Parameter(openvino._pyopenvino.Node):
    """
    openvino.op.Parameter wraps ov::op::v0::Parameter
    """
    element_type: openvino._pyopenvino.Type
    layout: openvino._pyopenvino.Layout
    partial_shape: openvino._pyopenvino.PartialShape
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.Shape) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: openvino._pyopenvino.Type, arg1: openvino._pyopenvino.PartialShape) -> None:
        ...
    @typing.overload
    def __repr__(self: openvino._pyopenvino.Node) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def get_element_type(self) -> openvino._pyopenvino.Type:
        ...
    def get_layout(self) -> openvino._pyopenvino.Layout:
        ...
    @typing.overload
    def get_partial_shape(self) -> openvino._pyopenvino.PartialShape:
        ...
    @typing.overload
    def get_partial_shape(self) -> openvino._pyopenvino.PartialShape:
        ...
    def set_element_type(self, element_type: openvino._pyopenvino.Type) -> None:
        ...
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> None:
        ...
    def set_partial_shape(self, partial_shape: openvino._pyopenvino.PartialShape) -> None:
        ...
class Result(openvino._pyopenvino.Node):
    """
    openvino.op.Result wraps ov::op::v0::Result
    """
    layout: openvino._pyopenvino.Layout
    def __init__(self, arg0: openvino._pyopenvino.Output) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_layout(self) -> openvino._pyopenvino.Layout:
        ...
    def get_output_element_type(self, index: typing.SupportsInt) -> openvino._pyopenvino.Type:
        ...
    def get_output_partial_shape(self, index: typing.SupportsInt) -> openvino._pyopenvino.PartialShape:
        ...
    def set_layout(self, layout: openvino._pyopenvino.Layout) -> None:
        ...
class _PagedAttentionExtension(openvino._pyopenvino.Node):
    """
    Experimental extention for PagedAttention operation. Use with care: no backward compatibility is guaranteed in future releases.
    """
    def __init__(self, arg0: collections.abc.Sequence[openvino._pyopenvino.Output]) -> None:
        ...
class assign(openvino._pyopenvino.Node):
    """
    openvino.op.assign wraps ov::op::v6::Assign
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, new_value: typing.Any, variable: util.Variable, name: str = '') -> None:
        ...
    @typing.overload
    def __init__(self, new_value: typing.Any, variable_id: str, name: str = '') -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_variable_id(self) -> str:
        """
                    Gets variable id.
        
                    :return: variable id.
                    :rtype: str
        """
class if_op(openvino._pyopenvino.Node):
    """
    openvino.impl.op.If wraps ov::op::v0::If
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, execution_condition: openvino._pyopenvino.Output) -> None:
        """
                    Constructs If with condition.
        
                    :param execution_condition: condition node.
                    :type execution_condition: openvino.Output
        
                    :rtype: openvino.impl.op.If
        """
    @typing.overload
    def __init__(self, execution_condition: openvino._pyopenvino.Node) -> None:
        """
                    Constructs If with condition.
        
                    :param execution_condition: condition node.
                    :type execution_condition: openvino.Node
        
                    :rtype: openvino.impl.op.If
        """
    def __repr__(self) -> str:
        ...
    def get_else_body(self) -> typing.Any:
        """
                    Gets else_body as Model object.
        
                    :return: else_body as Model object.
                    :rtype: openvino.Model
        """
    def get_function(self, index: typing.SupportsInt) -> typing.Any:
        """
                    Gets internal sub-graph by index in MultiSubGraphOp.
        
                    :param index: sub-graph's index in op.
                    :type index: int
                    
                    :return: Model with sub-graph.
                    :rtype: openvino.Model
        """
    def get_input_descriptions(self, index: typing.SupportsInt) -> list:
        """
                    Gets list with connections between operation inputs and internal sub-graph parameters.
        
                    :param index: index of internal sub-graph.
                    :type index: int
        
                    :return: list of input descriptions.
                    :rtype: list[Union[openvino.op.util.MergedInputDescription,
                                       openvino.op.util.InvariantInputDescription,
                                       openvino.op.util.SliceInputDescription]]
        """
    def get_output_descriptions(self, index: typing.SupportsInt) -> list:
        """
                    Gets list with connections between operation outputs and internal sub-graph parameters.
        
                    :param index: index of internal sub-graph.
                    :type index: int
        
                    :return: list of output descriptions.
                    :rtype: list[Union[openvino.op.util.BodyOutputDescription,
                                      openvino.op.util.ConcatOutputDescription]]
        """
    def get_then_body(self) -> typing.Any:
        """
                    Gets then_body as Model object.
        
                    :return: then_body as Model object.
                    :rtype: openvino.Model
        """
    def set_else_body(self, body: typing.Any) -> None:
        """
                    Sets new Model object as new else_body.
        
                    :param body: new body for 'else' branch.
                    :type body: openvino.Model
        
                    :rtype: None
        """
    def set_function(self, index: typing.SupportsInt, func: typing.Any) -> None:
        """
                    Adds sub-graph to MultiSubGraphOp.
        
                    :param index: index of new sub-graph.
                    :type index: int
        
                    :param func: func new sub_graph as a Model.
                    :type func: openvino.Model
        
                    :rtype: None
        """
    def set_input(self, value: openvino._pyopenvino.Output, then_parameter: Parameter, else_parameter: Parameter) -> None:
        """
                    Sets new input to the operation associated with parameters of each sub-graphs.
        
                    :param value: input to operation.
                    :type value: openvino.Output
        
                    :param then_result: parameter for then_body or nullptr.
                    :type then_result: openvino.Node
        
                    :param else_result: parameter for else_body or nullptr.
                    :type else_result: openvino.Node
        
                    :rtype: None
        """
    def set_input_descriptions(self, index: typing.SupportsInt, inputs: list) -> None:
        """
                    Sets list with connections between operation inputs and internal sub-graph parameters.
        
                    :param index: index of internal sub-graph.
                    :type index: int
        
                    :param inputs: list of input descriptions.
                    :type inputs: list[Union[openvino.op.util.MergedInputDescription,
                                             openvino.op.util.InvariantInputDescription,
                                             openvino.op.util.SliceInputDescription]]
        
                    :rtype: None
        """
    def set_output(self, then_result: Result, else_result: Result) -> openvino._pyopenvino.Output:
        """
                    Sets new output from the operation associated with results of each sub-graphs.
        
                    :param then_result: result from then_body.
                    :type then_result: op.Result
        
                    :param else_result: result from else_body.
                    :type else_result: op.Result
        
                    :return: output from operation.
                    :rtype: openvino.Output
        """
    def set_output_descriptions(self, index: typing.SupportsInt, outputs: list) -> None:
        """
                    Sets list with connections between operation outputs and internal sub-graph parameters.
        
                    :param index: index of internal sub-graph.
                    :type index: int
        
                    :param outputs: list of output descriptions.
                    :type outputs: list[Union[openvino.op.util.BodyOutputDescription,
                                              openvino.op.util.ConcatOutputDescription]]
        
                    :rtype: None
        """
    def set_then_body(self, body: typing.Any) -> None:
        """
                    Sets new Model object as new then_body.
        
                    :param body: new body for 'then' branch.
                    :type body: openvino.Model
        
                    :rtype: None
        """
class loop(openvino._pyopenvino.Node):
    """
    openvino.impl.op.Loop wraps ov::op::v0::Loop
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, trip_count: openvino._pyopenvino.Output, execution_condition: openvino._pyopenvino.Output) -> None:
        ...
    @typing.overload
    def __init__(self, trip_count: openvino._pyopenvino.Node, execution_condition: openvino._pyopenvino.Node) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_concatenated_slices(self, value: openvino._pyopenvino.Output, start: typing.SupportsInt, stride: typing.SupportsInt, part_size: typing.SupportsInt, end: typing.SupportsInt, axis: typing.SupportsInt) -> openvino._pyopenvino.Output:
        ...
    def get_function(self) -> typing.Any:
        ...
    def get_input_descriptions(self) -> list:
        ...
    def get_iter_value(self, body_value: openvino._pyopenvino.Output, iteration: typing.SupportsInt = -1) -> openvino._pyopenvino.Output:
        ...
    def get_num_iterations(self) -> int:
        ...
    def get_output_descriptions(self) -> list:
        ...
    def get_special_body_ports(self) -> list:
        ...
    def set_function(self, func: typing.Any) -> None:
        ...
    def set_input_descriptions(self, inputs: list) -> None:
        ...
    def set_invariant_input(self, body_parameter: Parameter, value: openvino._pyopenvino.Output) -> None:
        ...
    def set_merged_input(self, body_parameter: Parameter, initial_value: openvino._pyopenvino.Output, successive_value: openvino._pyopenvino.Output) -> None:
        ...
    def set_output_descriptions(self, outputs: list) -> None:
        ...
    def set_sliced_input(self, parameter: Parameter, value: openvino._pyopenvino.Output, start: typing.SupportsInt, stride: typing.SupportsInt, part_size: typing.SupportsInt, end: typing.SupportsInt, axis: typing.SupportsInt) -> None:
        ...
    def set_special_body_ports(self, special_body_ports: list) -> None:
        ...
class read_value(openvino._pyopenvino.Node):
    """
    openvino.op.read_value wraps ov::op::v6::ReadValue
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, new_value: typing.Any, variable: util.Variable) -> None:
        ...
    @typing.overload
    def __init__(self, variable: util.Variable) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_variable_id(self) -> str:
        """
                    Gets variable id.
        
                    :return: variable id.
                    :rtype: str
        """
class tensor_iterator(openvino._pyopenvino.Node):
    """
    openvino.impl.op.TensorIterator wraps ov::op::v0::TensorIterator
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_body(self) -> typing.Any:
        ...
    def get_concatenated_slices(self, value: openvino._pyopenvino.Output, start: typing.SupportsInt, stride: typing.SupportsInt, part_size: typing.SupportsInt, end: typing.SupportsInt, axis: typing.SupportsInt) -> openvino._pyopenvino.Output:
        ...
    def get_function(self) -> typing.Any:
        ...
    def get_input_descriptions(self) -> list:
        ...
    def get_iter_value(self, body_value: openvino._pyopenvino.Output, iteration: typing.SupportsInt = -1) -> openvino._pyopenvino.Output:
        ...
    def get_num_iterations(self) -> int:
        ...
    def get_output_descriptions(self) -> list:
        ...
    def set_body(self, body: typing.Any) -> None:
        ...
    def set_function(self, func: typing.Any) -> None:
        ...
    def set_input_descriptions(self, inputs: list) -> None:
        ...
    def set_invariant_input(self, body_parameter: Parameter, value: openvino._pyopenvino.Output) -> None:
        ...
    def set_merged_input(self, body_parameter: Parameter, initial_value: openvino._pyopenvino.Output, successive_value: openvino._pyopenvino.Output) -> None:
        ...
    def set_output_descriptions(self, outputs: list) -> None:
        ...
    def set_sliced_input(self, parameter: Parameter, value: openvino._pyopenvino.Output, start: typing.SupportsInt, stride: typing.SupportsInt, part_size: typing.SupportsInt, end: typing.SupportsInt, axis: typing.SupportsInt) -> None:
        ...
