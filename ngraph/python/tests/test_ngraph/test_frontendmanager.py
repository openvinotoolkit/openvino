# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager

def test_frontendmanager():
    fem = None
    try:
        fem = FrontEndManager()
    except Exception:
        assert False

    frontEnds = fem.availableFrontEnds()
    assert frontEnds is not None
