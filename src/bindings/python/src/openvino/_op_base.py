# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Optional, Tuple, List

from openvino._pyopenvino import Op as OpBase
from openvino._pyopenvino import Node, Output


class Op(OpBase):
    def __init__(self, py_obj: "Op", inputs: Optional[Union[List[Union[Node, Output]], Tuple[Union[Node, Output, List[Union[Node, Output]]]]]] = None) -> None:
        super().__init__(py_obj)
        self._update_type_info()
        if isinstance(inputs, tuple):
            inputs = None if len(inputs) == 0 else list(inputs)
            if inputs is not None and len(inputs) == 1 and isinstance(inputs[0], list):
                inputs = inputs[0]
        if inputs is not None:
            self.set_arguments(inputs)
            self.constructor_validate_and_infer_types()
