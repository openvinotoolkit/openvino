# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils import class_registration
from mo.utils.replacement_pattern import ReplacementPattern


class BackReplacementPattern(ReplacementPattern):
    registered_ops = {}
    registered_cls = []

    def run_after(self):
        from extensions.back.pass_separator import BackStart
        return [BackStart]

    def run_before(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.BACK_REPLACER


ReplacementPattern.excluded_replacers.append(BackReplacementPattern)
