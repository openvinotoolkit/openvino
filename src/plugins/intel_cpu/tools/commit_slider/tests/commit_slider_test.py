# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest import TestCase
from tests import skip_commit_slider_devtest

sys.path.append('./')
from test_util import getExpectedCommit
from test_util import getActualCommit
from test_util import checkBmStability
from test_data import FirstBadVersionData, FirstValidVersionData, BmStableData


class CommitSliderTest(TestCase):
    @skip_commit_slider_devtest
    def testFirstValidVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstValidVersionData())
        actualCommit = getActualCommit(updatedData)

        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testFirstBadVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstBadVersionData())
        actualCommit = getActualCommit(updatedData)

        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBmStability(self):
        isStable = checkBmStability(BmStableData())

        self.assertTrue(isStable)
