# type: ignore
from __future__ import annotations
from collections.abc import Callable
from openvino._op_base import Op
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import TensorVector
from openvino._pyopenvino import Type
from typing import cast
from typing import overload
import collections.abc
import openvino._op_base
import openvino._pyopenvino
__all__: list[str] = ['Callable', 'Op', 'PartialShape', 'PostponedConstant', 'Shape', 'Tensor', 'TensorVector', 'Type', 'cast', 'make_postponed_constant', 'overload']
class PostponedConstant(openvino._op_base.Op):
    """
    Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose.
    """
    def __init__(self, element_type: openvino._pyopenvino.Type, shape: openvino._pyopenvino.Shape, maker: collections.abc.Callable, name: typing.Optional[str] = None) -> None:
        """
        Creates a PostponedConstant.
        
                :param element_type: Element type of the constant.
                :type element_type: openvino.Type
                :param shape: Shape of the constant.
                :type shape: openvino.Shape
                :param maker: A callable that returns a Tensor or modifies the provided Tensor to represent the constant.
                            Note: It's recommended to use a callable without arguments (returns Tensor) to avoid unnecessary tensor data copies.
                :type maker: Union[Callable[[], Tensor], Callable[[Tensor], None]]
                :param name: Optional name for the constant.
                :type name: Optional[str]
        
                :Example of a maker that returns a Tensor:
        
                .. code-block:: python
        
                    class Maker:
                        def __call__(self) -> ov.Tensor:
                            tensor_data = np.array([2, 2, 2, 2], dtype=np.float32)
                            return ov.Tensor(tensor_data)
                
        """
    def clone_with_new_inputs(self, new_inputs: list[openvino._pyopenvino.Tensor]) -> openvino._op_base.Op:
        ...
    def evaluate(self, outputs: openvino._pyopenvino.TensorVector, _: list[openvino._pyopenvino.Tensor]) -> bool:
        ...
    def has_evaluate(self) -> bool:
        ...
    def validate_and_infer_types(self) -> None:
        ...
def make_postponed_constant(element_type: openvino._pyopenvino.Type, shape: openvino._pyopenvino.Shape, maker: collections.abc.Callable, name: typing.Optional[str] = None) -> openvino._op_base.Op:
    ...
