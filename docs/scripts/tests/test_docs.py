# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for Doxygen based documentation generation.
Refer to conftest.py on the test usage.
"""


def test_documentation_page(errors):
    """ Test documentation page has no errors generating
    """
    if errors:
        assert False, '\n'.join(['documentation has issues:'] +
                                sorted(errors))
