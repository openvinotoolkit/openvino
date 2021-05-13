# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager
from ngraph import CheckFailureFrontEnd
from ngraph import ErrorCode

def test_frontendmanager():
    fem = None
    try:
        fem = FrontEndManager()
    except Exception:
        assert False

    frontEnds = fem.availableFrontEnds()
    assert frontEnds is not None

    assert not("UnknownFramework" in frontEnds)
    try:
        fem.loadByFramework("UnknownFramework")
    except CheckFailureFrontEnd as exc:
        assert exc.ERROR_CODE == ErrorCode.INITIALIZATION_ERROR
