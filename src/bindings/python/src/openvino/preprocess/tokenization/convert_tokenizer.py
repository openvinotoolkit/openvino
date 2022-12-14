# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Optional

from openvino.runtime.exceptions import OVTypeError
from .tokenizer_pipeline import TokenizerPipeline



def convert_tokenizer(tokenizer_object: Any, number_of_inputs: Optional[int] = None) -> TokenizerPipeline:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from .huggingface_parser import TransformersTokenizerPipelineParser

        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            if number_of_inputs is not None:
                return TransformersTokenizerPipelineParser(tokenizer_object).parse(number_of_inputs=number_of_inputs)
            else:
                return TransformersTokenizerPipelineParser(tokenizer_object).parse()
    if "tensorflow" in sys.modules and "tensorflow_text" in sys.modules:
        from .tf_text_parser import TFTextParser

        return TFTextParser(tokenizer_object).parse()


    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")
