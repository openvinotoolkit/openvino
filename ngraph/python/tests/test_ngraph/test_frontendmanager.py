# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ngraph import FrontEndManager
from ngraph import InitializationFailure


def test_frontendmanager():
    fem = None
    try:
        fem = FrontEndManager()
    except Exception:
        assert False

    frontEnds = fem.get_available_front_ends()
    assert frontEnds is not None

    assert not("UnknownFramework" in frontEnds)
    try:
        fem.load_by_framework("UnknownFramework")
    except InitializationFailure as exc:
        print(exc)
    else:
        assert False
