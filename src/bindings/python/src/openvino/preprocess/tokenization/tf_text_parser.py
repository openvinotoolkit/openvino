# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Union, Optional, List

import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from .tokenizer_pipeline import (
    TokenizerPipeline,
    NormalizationStep,
    UnicodeNormalizationStep,
    LowercaseStep,
    RegExpNormalizationStep,
    StripStringStep,
    PreTokenizatinStep,
    PunctuationSplitStep,
    RegExpSplitStep,
    WhitespaceSplitStep,
    WordPieceTokenizationStep,
    AddTokenStep,
    SequenceStep,
    TruncationStep,
    AddPaddingStep,
    PostTokenizationStep,
)


@dataclass
class TFTextParser:
    tokenizer_object: Any
    graph: Optional[GraphDef] = None
    pipeline: TokenizerPipeline = field(default_factory=TokenizerPipeline, init=False)

    def __post_init__(self) -> None:
        self.graph = self.convert_to_frozen_graph(self.tokenizer_object)

    @staticmethod
    def convert_to_frozen_graph(tokenizer_object):
        concrete_func = tokenizer_object.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        frozen_func = convert_variables_to_constants_v2(
            concrete_func,
            lower_control_flow=True,
            aggressive_inlining=True,
        )
        return frozen_func.graph.as_graph_def(add_shapes=True)

    def _get_node_by_name(self, name) -> Optional[NodeDef]:
        return next(
            (node for node in self.graph.node if name in node.name),
            None,
        )

    def _get_nodes_by_name(self, name) -> List[NodeDef]:
        return [node for node in self.graph.node if name in node.name]

    def add_case_fold(self) -> None:
        fold_node = self._get_node_by_name(LowercaseStep.tf_node_name)
        if fold_node:
            self.pipeline.add_steps(LowercaseStep())

    def add_unicode_normalization(self) -> None:
        normalization_node = self._get_node_by_name(UnicodeNormalizationStep.tf_node_name)
        if normalization_node:
            self.pipeline.add_steps(
                UnicodeNormalizationStep(
                    normalization_form=normalization_node.attr["normalization_form"].s.decode()
                )
            )

    def add_regex_normalization(self) -> None:
        for node in self._get_nodes_by_name(RegExpNormalizationStep.tf_node_name):
            self.pipeline.add_steps(
                RegExpNormalizationStep(
                    regex_search_pattern=node.attr["pattern"].s.decode(),
                    replace_term=node.attr["rewrite"].s.decode(),
                )
            )

    def parse_regex_split(self) -> None:
        split_node = self._get_node_by_name(RegExpSplitStep.tf_node_name)
        if split_node:
            self.pipeline.add_steps(
                RegExpSplitStep(
                    split_pattern=split_node.attr["value"].tensor.string_val[0].decode()
                )
            )

    def parse(self) ->TokenizerPipeline:
        self.add_case_fold()
        self.add_unicode_normalization()
        self.add_regex_normalization()
        self.parse_regex_split()
        return self.pipeline
