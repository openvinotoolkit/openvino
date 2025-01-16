# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils import class_registration
from openvino.tools.mo.utils.replacement_pattern import ReplacementPattern


class BackReplacementPattern(ReplacementPattern):
    registered_ops = {}
    registered_cls = []

    def run_after(self):
        from openvino.tools.mo.back.pass_separator import BackStart
        return [BackStart]

    def run_before(self):
        from openvino.tools.mo.back.pass_separator import BackFinish
        return [BackFinish]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.BACK_REPLACER


ReplacementPattern.excluded_replacers.append(BackReplacementPattern)
