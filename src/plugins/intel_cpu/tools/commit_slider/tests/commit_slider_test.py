# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest import TestCase
from tests import skip_commit_slider_devtest

sys.path.append('./')
from test_util import getExpectedCommit
from test_util import getActualCommit
from utils.break_validator import validateBMOutput
from test_data import FirstBadVersionData, FirstValidVersionData,\
    BmStableData, BmValidatorSteppedBreakData, BmValidatorSteppedBreakData2
from utils.break_validator import BmValidationError


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
        td = BmStableData()
        isStable = validateBMOutput(
            td.bmOutputMap,
            td.breakCommit,
            td.dev)

        self.assertTrue(isStable)

    @skip_commit_slider_devtest
    def testBmSteppedBreak(self):
        td = BmValidatorSteppedBreakData()

        # wrong break commit with low deviation
        with self.assertRaises(BmValidationError) as e:
            validateBMOutput(
                td.bmOutputMap,
                td.wrongBreakCommit,
                td.highDev
            )
        self.assertEqual(
            str(e.exception),
            "pre-break interval does not majorize post-break"
        )

        # real break commit with decreased deviation
        isStable = validateBMOutput(
                td.bmOutputMap,
                td.realBreakCommit,
                td.lowDev
            )
        self.assertTrue(isStable)

    @skip_commit_slider_devtest
    def testBmSteppedBreak2(self):
        td = BmValidatorSteppedBreakData2()

        # local gap low, than expected
        with self.assertRaises(BmValidationError) as e:
            validateBMOutput(
                td.bmOutputMap,
                td.breakCommit,
                td.dev
            )
        self.assertEqual(
            e.exception.errType,
            BmValidationError.BmValErrType.LOW_LOCAL_GAP
        )

