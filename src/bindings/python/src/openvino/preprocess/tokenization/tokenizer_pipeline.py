# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Any, Dict
from openvino.runtime.exceptions import UserInputError


class BasePipelineStep:
    def __init__(self) -> None:
        self._params: Dict[str, Any] = {}
        self._pipeline: Optional["TokenizerPipeline"] = None

    def __setattr__(self, key: str, value: Any) -> None:
        if not key.startswith("_"):
            self._params[key] = value
        super().__setattr__(key, value)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self._params.items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        return self._params

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = pipeline


class NormalizationPipelineStep(BasePipelineStep):
    pass


class NFDNormalizationStep(NormalizationPipelineStep):
    pass


class LowercaseStep(NormalizationPipelineStep):
    pass


class RegExpNormalizationStep(NormalizationPipelineStep):
    def __init__(self, regex_search_pattern: str, replace_term: str) -> None:
        super().__init__()
        self.regex_search_pattern = regex_search_pattern
        self.replace_term = replace_term

    @classmethod
    def strip_accents_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def del_control_chars_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Cc}|\p{Cf}", replace_term=" ")


class StripAccentsStep(NormalizationPipelineStep):
    pass


class DelControlCharsStep(NormalizationPipelineStep):
    pass


class PreTokenizatinStep(BasePipelineStep):
    pass


class WhitespaceSplitStep(PreTokenizatinStep):
    pass


class RegExpSplitStep(PreTokenizatinStep):
    def __init__(self, split_pattern: str) -> None:
        super().__init__()
        self.split_pattern = split_pattern

    @classmethod
    def bert_splitter(cls) -> "RegExpSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"\s+",
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
        )


class TokenizationModelStep(BasePipelineStep):
    pass


class WordPieceTokenizationStep(TokenizationModelStep):
    def __init__(
        self,
        vocab: List[str],
        unk_token: str = "[UNK]",
        subword_prefix: str = "##",
        max_chars_per_word: int = 100,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.unk_token = unk_token
        try:
            self.unk_token_idx = self.vocab.index(unk_token)
        except ValueError:
            raise UserInputError(f"Unknown token {self.token} is not in vocab")
        self.subword_prefix = subword_prefix
        self.max_chars_per_word = max_chars_per_word

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self._params.items() if key != "vocab")
        return f"{self.__class__.__name__}({params_string})"

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            subword_prefix=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )


class PostTokenizationStep(BasePipelineStep):
    pass


class AddTokenStep(PostTokenizationStep):
    def __init__(self, token: str, token_type_id: Optional[int] = None) -> None:
        super().__init__()
        self.token = token

        self.token_type_id = token_type_id

        self.token_idx: Optional[int] = None
        self.set_token_idx()

    def set_token_idx(self) -> None:
        pipeline = self.get_pipeline()
        if pipeline is None or pipeline.vocab is None:
            return
        try:
            self.token_idx = pipeline.vocab.index(self.token)
        except ValueError:
            raise UserInputError(f"Special token {self.token} is not in vocab")


class SequenceStep(PostTokenizationStep):
    def __init__(self, token_type_id: Optional[int] = None):
        super().__init__()
        self.token_type_id = token_type_id


class AddPaddingStep(PostTokenizationStep):
    def __init__(self, padding_token: str, pad_right: bool = True) -> None:
        super().__init__()
        self.padding_token = padding_token
        self.pad_right = pad_right


class TruncationStep(PostTokenizationStep):
    def __init__(self, max_length: int, truncate_right: bool = True) -> None:
        super().__init__()
        self.max_length = max_length
        self.truncate_right = truncate_right

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any], num_of_added_tokens: int = 0) -> "TruncationStep":
        return cls(
            max_length=tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        return cls(
            max_length=tokenizer.model_max_length - num_of_added_tokens,
            truncate_right=tokenizer.truncation_side == "right",
        )


class TokenizerPipeline:
    def __init__(self, steps: Optional[List[BasePipelineStep]] = None) -> None:
        self.steps = steps or []
        self.vocab: Optional[List[str]] = None

    def generate_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    def add_step(self, step: BasePipelineStep) -> None:
        self.steps.append(step)
        step.set_pipeline(self)

    def __str__(self) -> str:
        steps = "\n\t".join(str(step) for step in self.steps)
        return f"TokenizerPipeline(\n\t{steps}\n)"

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]
