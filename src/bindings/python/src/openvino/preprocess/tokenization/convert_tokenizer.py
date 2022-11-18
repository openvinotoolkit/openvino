# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any

from openvino.runtime.exceptions import OVTypeError
from .tokenizer_pipeline import TokenizerPipeline
from .parsers import TransformersTokenizerPipelineParser


def convert_tokenizer(tokenizer_object: Any) -> TokenizerPipeline:
    """Converts a tokenizer object to an OpenVINO tokenizer.

    Supported frameworks:
    - Transformers

    Example:
    >>> from openvino.preprocess import convert_tokenizer
    >>> from transformers import AutoTokenizer
    >>> hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> ov_tokenizer = convert_tokenizer(hf_tokenizer)

    :param tokenizer_object: tokenizer object from supported framework
    :type tokenizer_object: Any
    :return: Tokenizer pipeline config [todo: OpenVINO tokenizer object]
    :rtype: TokenizerPipeline
    """
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase

        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            return TransformersTokenizerPipelineParser(tokenizer_object).parse()

    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")
