# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils import class_registration
from mo.utils.replacement_pattern import ReplacementPattern


class MiddleReplacementPattern(ReplacementPattern):
    registered_ops = {}
    registered_cls = []

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.MIDDLE_REPLACER


ReplacementPattern.excluded_replacers.append(MiddleReplacementPattern)
