# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.utils import deprecatedclassproperty

from openvino._pyopenvino.properties.hint import PerformanceMode as PerformanceModeBase


class PerformanceMode(PerformanceModeBase):

    @deprecatedclassproperty(
        name="PerformanceMode.UNDEFINED",  # noqa: N802, N805
        version="2024.0",
        message="Please use actual value instead.",
        stacklevel=2,
    )
    def UNDEFINED(cls) -> PerformanceModeBase:  # noqa: N802, N805
        return super().UNDEFINED
