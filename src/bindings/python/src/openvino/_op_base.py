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
            # inputs = None if len(inputs) == 0 else list(inputs)
            if len(inputs) == 0:
                inputs = None
            if inputs is not None and len(inputs) == 1 and isinstance(inputs[0], list):
                inputs = inputs[0]

        if isinstance(inputs,list):
            flat_list = []
            for items in inputs:
                if isinstance(items,list):
                    flat_list.extend(items)
                else:
                    flat_list.append(items)
            inputs = flat_list

        if inputs is not None:
            output_list = [item for item in inputs if isinstance(item, Output)]
            self.set_arguments(output_list)
            self.constructor_validate_and_infer_types()
