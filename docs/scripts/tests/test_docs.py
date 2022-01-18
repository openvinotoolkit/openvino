# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for Doxygen based documentation generation.
Refer to conftest.py on the test usage.
"""


def test_documentation_page(doxygen_errors):
    """ Test documentation page has no errors generating
    """
    if doxygen_errors:
        assert False, '\n'.join(['documentation has issues:'] +
                                sorted(doxygen_errors))
