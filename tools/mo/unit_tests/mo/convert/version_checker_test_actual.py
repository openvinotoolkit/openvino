# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.runtime import get_version as get_ie_version

from openvino.tools.mo.utils.version import get_version, get_simplified_ie_version, \
    get_simplified_mo_version, VersionChecker


class VersionCheckerTest(unittest.TestCase):
    def test_version_checker(self):
        import datetime
        import os
        ref_mo_version = get_version()
        ref_ie_version = get_ie_version()
        ref_mo_simplified_version = get_simplified_mo_version()
        ref_ie_simplified_version = get_simplified_ie_version(env=os.environ)

        # first init of VersionChecker
        start_time = datetime.datetime.now()
        VersionChecker().check_runtime_dependencies()
        VersionChecker().get_mo_version()
        VersionChecker().get_ie_version()
        VersionChecker().get_mo_simplified_version()
        VersionChecker().get_ie_simplified_version()
        first_init_time = (datetime.datetime.now() - start_time).total_seconds()

        # Loop with multiple usages of VersionChecker
        start_time = datetime.datetime.now()
        for _ in range(100):
            VersionChecker().check_runtime_dependencies()
            assert VersionChecker().get_mo_version() == ref_mo_version
            assert VersionChecker().get_ie_version() == ref_ie_version
            assert VersionChecker().get_mo_simplified_version() == ref_mo_simplified_version
            assert VersionChecker().get_ie_simplified_version() == ref_ie_simplified_version
        loop_time = (datetime.datetime.now() - start_time).total_seconds()

        # Check that time of loop is less than first init, so no actual initialization happens
        assert loop_time < first_init_time
