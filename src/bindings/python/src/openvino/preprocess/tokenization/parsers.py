# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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
    TruncationStep,
)


class TransformersTokenizerPipelineParser:
    def __init__(self, tokenizer_object: Any) -> None:
        self.original_tokenizer = tokenizer_object
        with TemporaryDirectory() as tmpdir:
            tokenizer_object.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "tokenizer.json") as tj:
                self.tokenizer_json = json.load(tj)
        self.pipeline = TokenizerPipeline()

    def parse(self) -> TokenizerPipeline:
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
            vocab = [token for token, index in sorted(self.tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])]
            self.pipeline.add_step(
                WordPieceTokenizationStep(
                    unk_token=self.tokenizer_json["model"]["unk_token"],
                    subword_prefix=self.tokenizer_json["model"]["continuing_subword_prefix"],
                    vocab=vocab,
                ),
            )
            self.pipeline.vocab = vocab
        else:
            raise OVTypeError(f'Tokenizer type :{self.tokenizer_json["model"]["type"]} is not supported')

    def post_tokenization(self) -> None:
        num_of_added_tokens = 0
        add_tokens_steps = []
        if "single" in self.tokenizer_json["post_processor"]:
            insert_first = True
            for template_dict in self.tokenizer_json["post_processor"]["single"]:
                if "SpecialToken" in template_dict:
                    num_of_added_tokens += 1
                    add_tokens_steps.append(
                        AddTokenStep(
                            token=template_dict["SpecialToken"]["id"],
                            insert_first=insert_first,
                        ),
                    )
                else:
                    # current template dict is for sequence
                    insert_first = False
        if self.original_tokenizer.model_max_length is not None:
            self.pipeline.add_step(
                TruncationStep(
                    max_length=self.original_tokenizer.model_max_length - num_of_added_tokens,
                    truncate_right=self.original_tokenizer.truncation_side == "right",
                ),
            )

        for step in add_tokens_steps:
            self.pipeline.add_step(step)
            step.set_token_idx()

    parsing_pipeline = [
        nfd_normalization,
        do_lowercase,
        strip_accents,
        del_control_chars,
        pre_tokenization,
        tokenization_model,
        post_tokenization,
    ]
