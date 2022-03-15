# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.pyopenvino.passes import Manager as ManagerBase


class Manager(ManagerBase):
    """Manager that additionally holds transformations objects."""
    def __init__(self):
        super().__init__()
        self.passes_list = []  # need to keep python instances alive

    def register_pass(self, transformation):
        self.passes_list.append(transformation)
        return super().register_pass(transformation)
