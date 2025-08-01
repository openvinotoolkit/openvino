# type: ignore
from __future__ import annotations
import collections.abc
import openvino._pyopenvino
import openvino._pyopenvino.op
import typing
"""
Package openvino.passes wraps ov::passes
"""
__all__ = ['AnyInput', 'BackwardGraphRewrite', 'ConstantFolding', 'ConvertFP32ToFP16', 'GraphRewrite', 'LowLatency2', 'MakeStateful', 'Manager', 'Matcher', 'MatcherPass', 'ModelPass', 'Optional', 'Or', 'PassBase', 'PatternSymbolValue', 'Predicate', 'Serialize', 'Version', 'VisualizeTree', 'WrapType', 'attrs_match', 'consumers_count', 'has_static_dim', 'has_static_dims', 'has_static_rank', 'has_static_shape', 'rank_equals', 'rank_more_than', 'shape_matches', 'type_matches', 'type_matches_any']
class AnyInput(openvino._pyopenvino.Node):
    """
    openvino.passes.AnyInput wraps ov::pass::pattern::op::Label
    """
    @typing.overload
    def __init__(self) -> None:
        """
                          Create pattern AnyInput operation which is used to match any type of node.
        """
    @typing.overload
    def __init__(self, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create pattern AnyInput operation which is used to match any type of node.
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: function
        """
    @typing.overload
    def __init__(self, predicate: Predicate) -> None:
        """
                          Create pattern AnyInput operation which is used to match any type of node.
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: function
        """
    def __repr__(self) -> str:
        ...
class BackwardGraphRewrite(GraphRewrite, ModelPass, PassBase):
    """
    openvino.passes.BackwardGraphRewrite executes sequence of MatcherPass transformations in reversed topological order
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, matcher_pass: MatcherPass) -> None:
        """
                                   Register single MatcherPass pass inside BackwardGraphRewrite.
        
                                   :param pass: openvino.passes.MatcherPass instance
                                   :type pass: openvino.passes.MatcherPass
        """
    def __repr__(self) -> str:
        ...
    def add_matcher(self, matcher_pass: MatcherPass) -> MatcherPass:
        """
                Register single MatcherPass pass inside BackwardGraphRewrite.
        
                :param pass: openvino.passes.MatcherPass instance
                :type pass: openvino.passes.MatcherPass
        """
class ConstantFolding(ModelPass, PassBase):
    """
    openvino.passes.ConstantFolding transformation
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class ConvertFP32ToFP16(ModelPass, PassBase):
    """
    openvino.passes.ConvertFP32ToFP16 transformation
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
class GraphRewrite(ModelPass, PassBase):
    """
    openvino.passes.GraphRewrite executes sequence of MatcherPass transformations in topological order
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, matcher_pass: MatcherPass) -> None:
        """
                              Register single MatcherPass pass inside GraphRewrite.
        
                              :param pass: openvino.passes.MatcherPass instance
                              :type pass: openvino.passes.MatcherPass
        """
    def add_matcher(self, matcher_pass: MatcherPass) -> MatcherPass:
        """
                              Register single MatcherPass pass inside GraphRewrite.
        
                              :param pass: openvino.passes.MatcherPass instance
                              :type pass: openvino.passes.MatcherPass
        """
class LowLatency2(ModelPass, PassBase):
    """
    openvino.passes.LowLatency2 transformation
    """
    def __init__(self, use_const_initializer: bool = True) -> None:
        """
                            Create LowLatency2 pass which is used for changing the structure of the model,
                            which contains TensorIterator/Loop operations.
                            The transformation finds all TensorIterator/Loop layers in the network, 
                            processes all back edges that describe a connection between Result and Parameter of the TensorIterator/Loop bodies, 
                            and inserts ReadValue and Assign layers at the input and output corresponding to this back edge.
        
                            :param use_const_initializer: Changes the type of the initializing subgraph for ReadValue operations.
                                                          If "true", then the transformation inserts Constant before ReadValue operation.
                                                          If "false, then the transformation leaves existed initializing subgraph for ReadValue operation.
                            :type use_const_initializer: bool
        """
    def __repr__(self) -> str:
        ...
class MakeStateful(ModelPass, PassBase):
    """
    openvino.passes.MakeStateful transformation
    """
    @typing.overload
    def __init__(self, pairs_to_replace: collections.abc.Sequence[tuple[openvino._pyopenvino.op.Parameter, openvino._pyopenvino.op.Result]]) -> None:
        """
         The transformation replaces the provided pairs Parameter and Result with openvino Memory operations ReadValue and Assign.
                            
                              :param pairs_to_replace:
                              :type pairs_to_replace: list[tuple[op.Parameter, op.Result]
        """
    @typing.overload
    def __init__(self, pairs_to_replace: collections.abc.Mapping[str, str]) -> None:
        """
                The transformation replaces the provided pairs Parameter and Result with openvino Memory operations ReadValue and Assign.
                
                :param pairs_to_replace: a dictionary of names of the provided Parameter and Result operations.
                :type pairs_to_replace: dict[str, str]
        """
    def __repr__(self) -> str:
        ...
class Manager:
    """
    openvino.passes.Manager executes sequence of transformation on a given Model
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def register_pass(self, transformation: PassBase) -> PassBase:
        """
                        Register pass instance for execution. Execution order matches the registration order.
        
                        :param transformation: transformation instance.
                        :type transformation: openvino.passes.PassBase
        """
    def run_passes(self, model: typing.Any) -> None:
        """
                        Executes sequence of transformations on given Model.
        
                        :param model: openvino.Model to be transformed.
                        :type model: openvino.Model
        """
    def set_per_pass_validation(self, new_state: bool) -> None:
        """
                        Enables or disables Model validation after each pass execution.
        
                        :param new_state: flag which enables or disables model validation.
                        :type new_state: bool
        """
class Matcher:
    """
    openvino.passes.Matcher wraps ov::pass::pattern::Matcher
    """
    @typing.overload
    def __init__(self, node: openvino._pyopenvino.Node, name: str) -> None:
        """
                        Creates Matcher object with given pattern root node and matcher name.
                        Matcher object is used for pattern matching on Model.
        
                        :param node: pattern root node.
                        :type node: openvino.Node
        
                        :param name: pattern name. Usually matches the MatcherPass class name.
                        :type name: str
        """
    @typing.overload
    def __init__(self, output: openvino._pyopenvino.Output, name: str) -> None:
        """
                        Creates Matcher object with given pattern root node output and matcher name.
                        Matcher object is used for pattern matching on Model.
        
                        :param node: pattern root node output.
                        :type node: openvino.Output
        
                        :param name: pattern name. Usually matches the MatcherPass class name.
                        :type name: str
        """
    def get_match_nodes(self) -> list[openvino._pyopenvino.Node]:
        """
                        Get NodeVector of matched nodes. Should be used after match() method is called.
        
                        :return: matched nodes vector.
                        :rtype: list[openvino.Node]
        """
    def get_match_root(self) -> openvino._pyopenvino.Node:
        """
                        Get matched root node inside Model. Should be used after match() method is called.
        
                        :return: matched node.
                        :rtype: openvino.Node
        """
    def get_match_value(self) -> openvino._pyopenvino.Output:
        """
                        Get matched node output inside Model. Should be used after match() method is called.
        
                        :return: matched node output.
                        :rtype: openvino.Output
        """
    def get_match_values(self) -> list[openvino._pyopenvino.Output]:
        """
                        Get OutputVector of matched outputs. Should be used after match() method is called.
        
                        :return: matched outputs vector.
                        :rtype: list[openvino.Output]
        """
    def get_name(self) -> str:
        """
                        Get Matcher name.
        
                        :return: openvino.passes.Matcher name.
                        :rtype: str
        """
    def get_pattern_value_map(self) -> dict[openvino._pyopenvino.Node, openvino._pyopenvino.Output]:
        """
                        Get map which can be used to access matched nodes using nodes from pattern.
                        Should be used after match() method is called.
        
                        :return: mapping of pattern nodes to matched nodes.
                        :rtype: dict
        """
    def get_symbols(self) -> typing.Any:
        """
                        Get map which can be used to access matched symbols using nodes from pattern.
                        Should be used after match() method is called.
        
                        :return: mapping of symbol names to symbol values.
                        :rtype: Any
        """
    @typing.overload
    def match(self, arg0: openvino._pyopenvino.Output) -> bool:
        """
                        Matches registered pattern starting from given output.
        
                        :return: status of matching.
                        :rtype: bool
        """
    @typing.overload
    def match(self, arg0: openvino._pyopenvino.Node) -> bool:
        """
                        Matches registered pattern starting from given Node.
        
                        :return: status of matching.
                        :rtype: bool
        """
class MatcherPass(PassBase):
    """
    openvino.passes.MatcherPass wraps ov::pass::MatcherPass
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, matcher: Matcher, callback: collections.abc.Callable[[Matcher], bool]) -> None:
        """
                Create MatcherPass from existing Matcher and callback objects.
        
                :param matcher: openvino.passes.Matcher with registered pattern.
                :type matcher: openvino.passes.Matcher
        
                :param callback: Function that performs transformation on the matched nodes.
                :type callback: function
        
                :return: created openvino.passes.MatcherPass instance.
                :rtype: openvino.passes.MatcherPass
        """
    def __repr__(self) -> str:
        ...
    def apply(self, node: openvino._pyopenvino.Node) -> bool:
        """
                             Execute MatcherPass on given Node.
        
                             :return: callback return code.
                             :rtype: bool
        """
    def register_matcher(self, matcher: Matcher, callback: collections.abc.Callable[[Matcher], bool]) -> None:
        """
                             Initialize matcher and callback for further execution.
        
                             :param matcher: openvino.passes.Matcher with registered pattern.
                             :type matcher: openvino.passes.Matcher
        
                             :param callback: Function that performs transformation on the matched nodes.
                             :type callback: function
        """
    def register_new_node(self, node: openvino._pyopenvino.Node) -> openvino._pyopenvino.Node:
        """
                             Register node for additional pattern matching.
        
                             :param node: openvino.Node for matching.
                             :type node: openvino.Node
        
                             :return: registered node instance
                             :rtype: openvino.Node
        """
class ModelPass(PassBase):
    """
    openvino.passes.ModelPass wraps ov::pass::ModelPass
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def run_on_model(self, model: typing.Any) -> None:
        """
                           run_on_model must be defined in inherited class. This method is used to work with Model directly.
        
                           :param model: openvino.Model to be transformed.
                           :type model: openvino.Model
        
                           :return: True in case if Model was changed and False otherwise.
                           :rtype: bool
        """
class Optional(openvino._pyopenvino.Node):
    """
    openvino.passes.Optional wraps ov::pass::pattern::op::Optional
    """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str]) -> None:
        """
                Create Optional with the given node type.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output) -> None:
        """
                Create Optional with the given node type and input node.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node's output.
                :type input: openvino.Output
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node.
                :type input: openvino.Node
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output]) -> None:
        """
                Create Optional with the given node type and input node.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node's output list.
                :type inputs: list[openvino.Output]
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node]) -> None:
        """
                Create Optional with the given node type and input node.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node list
                :type inputs: list[openvino.Node]
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create Optional with the given node type and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], predicate: Predicate) -> None:
        """
                Create Optional with the given node type and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node's output.
                :type input: openvino.Output
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output, predicate: Predicate) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node's output.
                :type input: openvino.Output
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node
                :type input: openvino.Node
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node, predicate: Predicate) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: input node
                :type input: openvino.Node
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node's output list.
                :type inputs: list[openvino.Output]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: Predicate) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node's output list.
                :type inputs: list[openvino.Output]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node list
                :type inputs: list[openvino.Node]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self: typing.Optional, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: Predicate) -> None:
        """
                Create Optional with the given node type, input node and predicate.
        
                :param type_names: node type. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: input node list
                :type inputs: list[openvino.Node]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    def __repr__(self: typing.Optional) -> str:
        ...
class Or(openvino._pyopenvino.Node):
    """
    openvino.passes.Or wraps ov::pass::pattern::op::Or
    """
    @typing.overload
    def __init__(self, inputs: collections.abc.Sequence[openvino._pyopenvino.Output]) -> None:
        """
                        Create pattern Or operation which is used to match any of given inputs.
        
                        :param inputs: Operation inputs.
                        :type inputs: list[openvino.Output]
        """
    @typing.overload
    def __init__(self, inputs: collections.abc.Sequence[openvino._pyopenvino.Node]) -> None:
        """
                        Create pattern Or operation which is used to match any of given inputs.
        
                        :param inputs: Operation inputs.
                        :type inputs: list[openvino.Node]
        """
    def __repr__(self) -> str:
        ...
class PassBase:
    """
    openvino.passes.PassBase wraps ov::pass::PassBase
    """
    def __repr__(self) -> str:
        ...
    def get_name(self) -> str:
        """
                          Get transformation name.
        
                          :return: Transformation name.
                          :rtype: str
        """
    def set_name(self, name: str) -> None:
        """
                          Set transformation name.
        
                          :param name: Transformation name.
                          :type name: str
        """
class PatternSymbolValue:
    """
    openvino.passes.PatternSymbolValue wraps ov::pass::pattern::PatternSymbolValue
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: PatternSymbolValue) -> bool:
        ...
    @typing.overload
    def __init__(self, value: openvino._pyopenvino.Symbol) -> None:
        """
                Create PatternSymbolValue with the given value.
        
                :param value: symbol to keep as pattern value
                :type value: openvino.Symbol
        """
    @typing.overload
    def __init__(self, value: typing.SupportsInt) -> None:
        """
                Create PatternSymbolValue with the given value.
        
                :param value: integer to keep as a pattern value
                :type value: int
        """
    @typing.overload
    def __init__(self, value: typing.SupportsFloat) -> None:
        """
                Create PatternSymbolValue with the given value.
        
                :param value: float to keep as a pattern value
                :type value: float
        """
    @typing.overload
    def __init__(self, value: collections.abc.Sequence[PatternSymbolValue]) -> None:
        """
                Create PatternSymbolValue with the given value.
        
                :param value: list of values representing a group of pattern values
                :type value: list[PatternSymbolValue]
        """
    def d(self) -> float:
        ...
    def g(self) -> list[PatternSymbolValue]:
        ...
    def i(self) -> int:
        ...
    def is_double(self) -> bool:
        ...
    def is_dynamic(self) -> bool:
        ...
    def is_group(self) -> bool:
        ...
    def is_integer(self) -> bool:
        ...
    def is_static(self) -> bool:
        ...
    def s(self) -> openvino._pyopenvino.Symbol:
        ...
class Predicate:
    """
    openvino.passes.Predicate wraps ov::pass::pattern::op::Predicate
    """
    @typing.overload
    def __init__(self) -> None:
        """
                          Create default Predicate which always returns true.
        """
    @typing.overload
    def __init__(self, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create Predicate from a given function.
        
                          :param predicate: function (Output<Node> -> bool)
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, predicate: collections.abc.Callable[[collections.abc.Mapping[str, ...], openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create Predicate from a given function.
        
                          :param predicate: function (PatternSymbolMap&, Output<Node> -> bool)
                          :type predicate: Callable
        """
class Serialize(ModelPass, PassBase):
    """
    openvino.passes.Serialize transformation
    """
    def __init__(self, path_to_xml: typing.Any, path_to_bin: typing.Any, version: typing.Any = None) -> None:
        """
                Create Serialize pass which is used for Model to IR serialization.
        
                :param path_to_xml: Path where *.xml file will be saved.
                :type path_to_xml: Union[str, bytes, pathlib.Path]
        
                :param path_to_xml: Path where *.bin file will be saved.
                :type path_to_xml: Union[str, bytes, pathlib.Path]
        
                :param version: Optional serialized IR version.
                :type version: Union[str, openvino.passes.Version]
        """
    def __repr__(self) -> str:
        ...
class Version:
    """
    Members:
    
      UNSPECIFIED
    
      IR_V10
    
      IR_V11
    """
    IR_V10: typing.ClassVar[Version]  # value = <Version.IR_V10: 10>
    IR_V11: typing.ClassVar[Version]  # value = <Version.IR_V11: 11>
    UNSPECIFIED: typing.ClassVar[Version]  # value = <Version.UNSPECIFIED: 0>
    __members__: typing.ClassVar[dict[str, Version]]  # value = {'UNSPECIFIED': <Version.UNSPECIFIED: 0>, 'IR_V10': <Version.IR_V10: 10>, 'IR_V11': <Version.IR_V11: 11>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class VisualizeTree(ModelPass, PassBase):
    """
    openvino.passes.VisualizeTree transformation
    """
    def __init__(self, file_name: str, nm: collections.abc.Callable[[openvino._pyopenvino.Node, collections.abc.Sequence[str]], None] = None, don_only: bool = False) -> None:
        """
                          Create VisualizeTree pass which is used for Model to dot serialization.
        
                          :param file_name: Path where serialized model will be saved. For example: /tmp/out.svg
                          :type file_name: str
        
                          :param nm: Node modifier function.
                          :type nm: function
        
                          :param don_only: Enable only dot file generation.
                          :type don_only: bool
        """
    def __repr__(self) -> str:
        ...
class WrapType(openvino._pyopenvino.Node):
    """
    openvino.passes.WrapType wraps ov::pass::pattern::op::WrapType
    """
    @typing.overload
    def __init__(self, type_name: str) -> None:
        """
                          Create WrapType with given node type.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        """
    @typing.overload
    def __init__(self, type_name: str, pred: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node type and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, pred: Predicate) -> None:
        """
                          Create WrapType with given node type and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Output) -> None:
        """
                          Create WrapType with given node type and input node.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param input: Node output.
                          :type input: openvino.Output
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Node) -> None:
        """
                          Create WrapType with given node type and input node.
        
                          :param type_name: node type. For example: opset8.Abs
                          :type type_name: str
        
                          :param input: Input node.
                          :type input: openvino.Node
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Output, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node type, input node and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param input: Node output.
                          :type input: openvino.Output
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Output, predicate: Predicate) -> None:
        """
                          Create WrapType with given node type, input node and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param input: Node output.
                          :type input: openvino.Output
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Node, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node type, input node and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param input: Input node.
                          :type input: openvino.Node
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, input: openvino._pyopenvino.Node, predicate: Predicate) -> None:
        """
                          Create WrapType with given node type, input node and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param input: Input node.
                          :type input: openvino.Node
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Output]) -> None:
        """
                          Create WrapType with given node type and input nodes.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Node outputs.
                          :type inputs: list[openvino.Output]
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Node]) -> None:
        """
                          Create WrapType with given node type and input nodes.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Input nodes.
                          :type inputs: list[openvino.Node]
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node type, input nodes and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Node outputs.
                          :type inputs: list[openvino.Output]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: Predicate) -> None:
        """
                          Create WrapType with given node type, input nodes and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Node outputs.
                          :type inputs: list[openvino.Output]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node type, input nodes and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Input nodes.
                          :type inputs: list[openvino.Node]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_name: str, inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: Predicate) -> None:
        """
                          Create WrapType with given node type, input nodes and predicate.
        
                          :param type_name: node type. For example: "opset8.Abs"
                          :type type_name: str
        
                          :param inputs: Input nodes.
                          :type inputs: list[openvino.Node]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str]) -> None:
        """
                          Create WrapType with given node types.
        
                          :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                          :type type_names: list[str]
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                          Create WrapType with given node types and predicate.
        
                          :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                          :type type_names: list[str]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], predicate: Predicate) -> None:
        """
                          Create WrapType with given node types and predicate.
        
                          :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                          :type type_names: list[str]
        
                          :param predicate: Function that performs additional checks for matching.
                          :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output) -> None:
        """
                          Create WrapType with given node types and input.
        
                          :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                          :type type_names: list[str]
        
                          :param input: Node output.
                          :type input: openvino.Output
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node) -> None:
        """
                          Create WrapType with given node types and input.
        
                          :param type_name: node types. For example: ["opset8.Abs", "opset8.Relu"]
                          :type type_name: list[str]
        
                          :param input: Input node.
                          :type input: openvino.Node
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create WrapType with given node types, input and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: Node output.
                :type input: openvino.Output
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Output, predicate: Predicate) -> None:
        """
                Create WrapType with given node types, input and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: Node output.
                :type input: openvino.Output
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node, predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create WrapType with given node types, input and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: Input node.
                :type input: openvino.Node
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], input: openvino._pyopenvino.Node, predicate: Predicate) -> None:
        """
                Create WrapType with given node types, input and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param input: Input node.
                :type input: openvino.Node
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output]) -> None:
        """
              Create WrapType with given node types and input.
        
              :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
              :type type_names: list[str]
        
              :param inputs: Nodes outputs.
              :type inputs: list[openvino.Output]
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node]) -> None:
        """
                Create WrapType with given node types and inputs.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: Input nodes.
                :type inputs: list[openvino.Node]
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create WrapType with given node types, inputs and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: Nodes outputs.
                :type inputs: list[openvino.Output]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Output], predicate: Predicate) -> None:
        """
                Create WrapType with given node types, inputs and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: Nodes outputs.
                :type inputs: list[openvino.Output]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: collections.abc.Callable[[openvino._pyopenvino.Output], bool]) -> None:
        """
                Create WrapType with given node types, inputs and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: Input nodes.
                :type inputs: list[openvino.Node]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    @typing.overload
    def __init__(self, type_names: collections.abc.Sequence[str], inputs: collections.abc.Sequence[openvino._pyopenvino.Node], predicate: Predicate) -> None:
        """
                Create WrapType with given node types, inputs and predicate.
        
                :param type_names: node types. For example: ["opset8.Abs", "opset8.Relu"]
                :type type_names: list[str]
        
                :param inputs: Input nodes.
                :type inputs: list[openvino.Node]
        
                :param predicate: Function that performs additional checks for matching.
                :type predicate: Callable
        """
    def __repr__(self) -> str:
        ...
def attrs_match(arg0: typing.Any) -> Predicate:
    ...
def consumers_count(arg0: typing.SupportsInt) -> Predicate:
    ...
def has_static_dim(arg0: typing.SupportsInt) -> Predicate:
    ...
def has_static_dims(arg0: collections.abc.Sequence[typing.SupportsInt]) -> Predicate:
    ...
def has_static_rank() -> Predicate:
    ...
def has_static_shape() -> Predicate:
    ...
def rank_equals(arg0: openvino._pyopenvino.Dimension) -> Predicate:
    ...
def rank_more_than(arg0: openvino._pyopenvino.Dimension) -> Predicate:
    ...
def shape_matches(arg0: str) -> Predicate:
    ...
def type_matches(arg0: openvino._pyopenvino.Type) -> Predicate:
    ...
def type_matches_any(arg0: collections.abc.Sequence[openvino._pyopenvino.Type]) -> Predicate:
    ...
