# type: ignore
from collections.abc import Mapping
from functools import singledispatchmethod
from __future__ import annotations
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import InferRequest as InferRequestBase
from openvino._pyopenvino import Tensor
import collections.abc
import numpy
import numpy as np
import openvino._pyopenvino
import typing
__all__ = ['ConstOutput', 'InferRequestBase', 'Mapping', 'OVDict', 'Tensor', 'np', 'singledispatchmethod', 'tensor_from_file']
class OVDict(collections.abc.Mapping):
    """
    Custom OpenVINO dictionary with inference results.
    
        This class is a dict-like object. It provides possibility to
        address data tensors with three key types:
    
        * `openvino.ConstOutput` - port of the output
        * `int` - index of the output
        * `str` - names of the output
    
        This class follows `frozenset`/`tuple` concept of immutability.
        It is prohibited to assign new items or edit them.
    
        To revert to the previous behavior use `to_dict` method which
        return shallow copy of underlaying dictionary.
        Note: It removes addressing feature! New dictionary keeps
              only `ConstOutput` keys.
    
        If a tuple returns value is needed, use `to_tuple` method which
        converts values to the tuple.
    
        :Example:
    
        .. code-block:: python
    
            # Reverts to the previous behavior of the native dict
            result = request.infer(inputs).to_dict()
            # or alternatively:
            result = dict(request.infer(inputs))
    
        .. code-block:: python
    
            # To dispatch outputs of multi-ouput inference:
            out1, out2, out3, _ = request.infer(inputs).values()
            # or alternatively:
            out1, out2, out3, _ = request.infer(inputs).to_tuple()
        
    """
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    @staticmethod
    def _OVDict__getitem_impl(*args, **kwargs) -> numpy.ndarray:
        ...
    def _(self, key: str) -> numpy.ndarray:
        ...
    def _OVDict__get_key(self, index: int) -> openvino._pyopenvino.ConstOutput:
        ...
    def _OVDict__get_names(self) -> typing.Dict[openvino._pyopenvino.ConstOutput, typing.Set[str]]:
        """
        Return names of every output key.
        
                Insert empty set if key has no name.
                
        """
    def __getitem__(self, key: typing.Union[openvino._pyopenvino.ConstOutput, int, str]) -> numpy.ndarray:
        ...
    def __init__(self, _dict: typing.Dict[openvino._pyopenvino.ConstOutput, numpy.ndarray]) -> None:
        ...
    def __iter__(self) -> typing.Iterator:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def items(self) -> typing.ItemsView[openvino._pyopenvino.ConstOutput, numpy.ndarray]:
        ...
    def keys(self) -> typing.KeysView[openvino._pyopenvino.ConstOutput]:
        ...
    def names(self) -> typing.Tuple[typing.Set[str], ...]:
        """
        Return names of every output key.
        
                Insert empty set if key has no name.
                
        """
    def to_dict(self) -> typing.Dict[openvino._pyopenvino.ConstOutput, numpy.ndarray]:
        """
        Return underlaying native dictionary.
        
                Function performs shallow copy, thus any modifications to
                returned values may affect this class as well.
                
        """
    def to_tuple(self) -> tuple:
        """
        Convert values of this dictionary to a tuple.
        """
    def values(self) -> typing.ValuesView[numpy.ndarray]:
        ...
class _InferRequestWrapper(openvino._pyopenvino.InferRequest):
    """
    InferRequest class with internal memory.
    """
    def __init__(self, other: openvino._pyopenvino.InferRequest) -> None:
        ...
    def _is_single_input(self) -> bool:
        ...
def tensor_from_file(path: str) -> openvino._pyopenvino.Tensor:
    """
    Create Tensor from file. Data will be read with dtype of unit8.
    """
