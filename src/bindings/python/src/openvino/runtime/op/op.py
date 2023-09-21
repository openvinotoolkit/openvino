# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._pyopenvino.op import if_op as if_op_base

from openvino.runtime import Model

class if_op(if_op_base):
    def __init__(self, if_op) -> None:
        super().__init__(if_op)
    
    def get_function(self, index):
        model = Model(super().get_function(index))
        print("OLOLO ", type(model))
        return model
     
    def get_else_body():
        return Model(super().get_else_body())