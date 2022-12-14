# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import List, Optional, Any, Dict, ClassVar

from openvino.runtime.exceptions import UserInputError, OVTypeError


@dataclass
class BasePipelineStep:
    _pipeline: Optional["TokenizerPipeline"] = field(default=None, init=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {key: getattr(self, key) for key in dir(type(self)) if not key.startswith("_") and isinstance(getattr(type(self), key), property)}
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = pipeline


@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class UnicodeNormalizationStep(NormalizationStep):
    normalization_form: str = "NFD"

    tf_node_name: ClassVar[str] = "NormalizeUTF8"


@dataclass
class NMTNormalizationStep(NormalizationStep):
    """Normaization based on NMT task.

    https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/tokenizers/src/normalizers/unicode.rs#L44
    """


@dataclass
class LowercaseStep(NormalizationStep):
    tf_node_name: ClassVar[str] = "CaseFoldUTF8"


@dataclass
class RegExpNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str

    tf_node_name: ClassVar[str] = "StaticRegexReplace"

    @classmethod
    def strip_accents_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def del_control_chars_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Cc}|\p{Cf}", replace_term=" ")


@dataclass
class StripAccentsStep(NormalizationStep):
    pass


@dataclass
class DelControlCharsStep(NormalizationStep):
    pass


@dataclass
class StripStringStep(NormalizationStep):
    left: bool
    right: bool


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""


@dataclass
class PunctuationSplitStep(PreTokenizatinStep):
    """Splits string on punctuation chars."""

    behaviour: str = "Isolated"


@dataclass
class RegExpSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "Remove"

    tf_node_name: ClassVar[str] = "RegexSplitWithOffsets/delim_regex_pattern"

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

    @classmethod
    def whitespace_splitter(cls) -> "RegExpSplitStep":
        return cls(r"\w+|[^\w\s]+")


@dataclass
class TokenizationModelStep(BasePipelineStep):
    pass


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: List[str]
    unk_token: str = "[UNK]"
    subword_prefix: str = "##"
    max_chars_per_word: int = 100
    unk_token_idx: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_idx = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items() if key != "vocab")
        return f"{self.__class__.__name__}({params_string})"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            subword_prefix=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class SpecialTokenWithIdx(PostTokenizationStep):
    token: str
    _token_idx: Optional[int] = None

    @property
    def token_idx(self) -> Optional[int]:
        if self._token_idx is not None:
            return self._token_idx

        pipeline = self.get_pipeline()
        if pipeline is None or pipeline.vocab is None:
            return None
        try:
            self._token_idx = pipeline.vocab.index(self.token)
        except ValueError:
            raise UserInputError(f"Special token {self.token} is not in vocab")

        return self._token_idx


@dataclass
class AddTokenStep(SpecialTokenWithIdx):
    token_type_id: Optional[int] = None


@dataclass
class SequenceStep(PostTokenizationStep):
    token_type_id: Optional[int]


@dataclass
class AddPaddingStep(SpecialTokenWithIdx):
    pad_right: bool = True
    token_type_id: Optional[int] = None

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "AddPaddingStep":
        padding_dict = tokenizer_json["padding"]
        return cls(
            token=padding_dict["pad_token"],
            pad_right=padding_dict["direction"] == "Right",
            token_type_id=padding_dict["pad_type_id"],
        )


@dataclass
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


@dataclass
class TokenizerPipeline:
    steps: List[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[List[str]] = None

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __str__(self) -> str:
        steps = "\n\t".join(str(step) for step in self.steps)
        return f"TokenizerPipeline(\n\t{steps}\n)"

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]
