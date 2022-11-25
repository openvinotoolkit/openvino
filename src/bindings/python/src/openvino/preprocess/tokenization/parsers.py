# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, List, Union

from openvino.runtime.exceptions import OVTypeError
from .tokenizer_pipeline import (
    TokenizerPipeline,
    NFDNormalizationStep,
    LowercaseStep,
    RegExpNormalizationStep,
    RegExpSplitStep,
    WhitespaceSplitStep,
    WordPieceTokenizationStep,
    AddTokenStep,
    SequenceStep,
    TruncationStep,
)


class TransformersTokenizerPipelineParser:
    def __init__(self, tokenizer_object: Any, number_of_inputs: int = 1) -> None:
        assert tokenizer_object.is_fast
        self.original_tokenizer = tokenizer_object
        with TemporaryDirectory() as tmpdir:
            tokenizer_object.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "tokenizer.json") as tj:
                self.tokenizer_json = json.load(tj)
        self.pipeline = TokenizerPipeline()
        self.number_of_inputs = number_of_inputs

    def parse(self, number_of_inputs: Optional[int] = None) -> TokenizerPipeline:
        self.number_of_inputs = self.number_of_inputs if number_of_inputs is None else number_of_inputs
        for add_step in self.parsing_pipeline:
            getattr(self, add_step.__name__)()

        return self.pipeline

    def nfd_normalization(self) -> None:
        if self.tokenizer_json["normalizer"]["type"] == "BertNormalizer":
            self.pipeline.add_step(NFDNormalizationStep())

    def do_lowercase(self) -> None:
        if self.tokenizer_json["normalizer"]["lowercase"] is True:
            self.pipeline.add_step(LowercaseStep())

    def del_control_chars(self) -> None:
        if self.tokenizer_json["normalizer"]["clean_text"] is True:
            self.pipeline.add_step(RegExpNormalizationStep.del_control_chars_regex())

    def strip_accents(self) -> None:
        if self.tokenizer_json["normalizer"]["strip_accents"] is True:
            self.pipeline.add_step(RegExpNormalizationStep.strip_accents_regex())

    def pre_tokenization(self) -> None:
        if self.tokenizer_json["pre_tokenizer"]["type"] == "BertPreTokenizer":
            self.pipeline.add_step(RegExpSplitStep.bert_splitter())
        else:
            self.pipeline.add_step(WhitespaceSplitStep())

    def tokenization_model(self) -> None:
        if self.tokenizer_json["model"]["type"] == "WordPiece":
            step = WordPieceTokenizationStep.from_hf_json(self.tokenizer_json)
            self.pipeline.add_step(step)
            self.pipeline.vocab = step.vocab
        else:
            raise OVTypeError(f'Tokenizer type :{self.tokenizer_json["model"]["type"]} is not supported')

    def post_tokenization(self) -> None:
        num_of_added_tokens = 0
        #  List contains two different types: AddTokenStep and SequenceStep
        #  Any other type declaration (or no type declaration) causes mypy error
        tokenizer_template_steps: List = []

        if self.number_of_inputs == 1:
            post_processor = self.tokenizer_json["post_processor"]["single"]
        else:
            post_processor = self.tokenizer_json["post_processor"]["pair"]

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                num_of_added_tokens += 1
                step = AddTokenStep(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                )
                step.set_pipeline(self.pipeline)
                step.set_token_idx()
                tokenizer_template_steps.append(step)
            else:
                tokenizer_template_steps.append(SequenceStep(token_type_id=template_dict["Sequence"]["type_id"]))

        if self.tokenizer_json["truncation"] is not None:
            self.pipeline.add_step(TruncationStep.from_hf_json(self.tokenizer_json, num_of_added_tokens))
        elif self.original_tokenizer.model_max_length is not None:
            self.pipeline.add_step(TruncationStep.from_hf_object(self.original_tokenizer, num_of_added_tokens))

        for step in tokenizer_template_steps:
            self.pipeline.add_step(step)

    parsing_pipeline = [
        nfd_normalization,
        do_lowercase,
        strip_accents,
        del_control_chars,
        pre_tokenization,
        tokenization_model,
        post_tokenization,
    ]
