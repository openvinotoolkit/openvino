# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest import TestCase
from tests import skip_commit_slider_devtest

sys.path.append('./')
from test_util import getExpectedCommit,\
    getBordersByTestData, getActualCommit
from utils.break_validator import validateBMOutput, BmValidationError
from test_data import FirstBadVersionData, FirstValidVersionData,\
    BmStableData, BmValidatorSteppedBreakData, BmValidatorSteppedBreakData2,\
    BenchmarkAppDataUnstable, BenchmarkAppDataStable, BenchmarkAppNoDegradationData,\
    BenchmarkAppUnstableDevData, BenchmarkAppWrongPathData, BenchmarkAppPathFoundData,\
    BenchmarkFirstFixedAppData, AcModeData, BenchmarkMetricData


class CommitSliderTest(TestCase):
    @skip_commit_slider_devtest
    def testFirstValidVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstValidVersionData())
        actualCommit, _ = getActualCommit(updatedData)

        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testFirstBadVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstBadVersionData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBmUnstable(self):
        _, updatedData = getExpectedCommit(
            BenchmarkAppDataUnstable())
        _, reason = getActualCommit(updatedData)
        self.assertEqual(reason, "left interval is stable, right interval is unstable")

    @skip_commit_slider_devtest
    def testBmStable(self):
        breakCommit, updatedData = getExpectedCommit(
            BenchmarkAppDataStable())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testACMode(self):
        breakCommit, updatedData = getExpectedCommit(
            AcModeData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBmFirstFixed(self):
        breakCommit, updatedData = getExpectedCommit(
            BenchmarkFirstFixedAppData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBmLatencyMetric(self):
        breakCommit, updatedData = getExpectedCommit(
            BenchmarkMetricData("latency:average"))
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBmNoDegradation(self):
        _, updatedData = getExpectedCommit(
            BenchmarkAppNoDegradationData())
        _, reason = getActualCommit(updatedData)
        reasonPrefix = reason.split(':')[0]
        self.assertEqual(reasonPrefix, "No degradation found")

    @skip_commit_slider_devtest
    def testBmUnstableDevice(self):
        _, updatedData = getExpectedCommit(
            BenchmarkAppUnstableDevData())
        _, reason = getActualCommit(updatedData)
        lCommit, rCommit = getBordersByTestData(updatedData)
        self.assertEqual(
            reason,
            "\"{}\" is unstable, \"{}\" is unstable".format(
                lCommit, rCommit
        ))

    @skip_commit_slider_devtest
    def testBmWrongPath(self):
        _, updatedData = getExpectedCommit(
            BenchmarkAppWrongPathData())
        _, reason = getActualCommit(updatedData)
        lCommit, rCommit = getBordersByTestData(updatedData)
        self.assertEqual(
            reason,
            "path /wrong/path.xml does not exist, check config"
        )

    @skip_commit_slider_devtest
    def testBmPathFound(self):
        expectedCommit, updatedData = getExpectedCommit(
            BenchmarkAppPathFoundData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(expectedCommit, actualCommit)

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

        # local gap lower, than expected
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

    @skip_commit_slider_devtest
    def testForMapSubstitutionRule(self):
        from utils.helpers import applySubstitutionRules
        cfg = {
                "serviceConfig": {
                    "previousKey": "previousValue"
                },
                "wrongDst": "{commitHash1} is unchanged",
                "dst": {
                    "complex": {
                        "path": [
                            "{commitHash1} is natural number",
                            "{commitHash2} is natural number",
                            "{commitHash1} is {commitHash2}"
                        ]
                    }
                },
                "src": {
                    "complex": {
                        "path": {
                            "one": "1",
                            "two": "2"
                        }
                    }
                }
        }
        rules = [
            {
                "name": "testRule1",
                "enabled": True,
                "type": "map",
                "placeholder": "commitHash1",
                "from": "$.src.complex.path",
                "to": "$.dst.complex.path"
            },
            {
                "name": "testRule2",
                "enabled": True,
                "type": "map",
                "placeholder": "commitHash2",
                "from": "$.src.complex.path",
                "to": "$.dst.complex.path"
            }
        ]
        def applyByRef(cfg: map, rules: list, substitution: str):
            applySubstitutionRules(cfg, rules, substitution)

        applyByRef(cfg, rules, "one")

        # assert first substitution
        self.assertEqual(
            cfg["dst"]["complex"]["path"][0],
            "1 is natural number"
        )
        self.assertEqual(
            cfg["dst"]["complex"]["path"][1],
            "1 is natural number"
        )
        self.assertEqual(
            cfg["dst"]["complex"]["path"][2],
            "1 is 1"
        )
        self.assertEqual(
            cfg["wrongDst"],
            "{commitHash1} is unchanged"
        )

        applyByRef(cfg, rules, "two")

        # assert second substitution
        self.assertEqual(
            cfg["dst"]["complex"]["path"][0],
            "2 is natural number"
        )
        self.assertEqual(
            cfg["dst"]["complex"]["path"][1],
            "2 is natural number"
        )
        self.assertEqual(
            cfg["dst"]["complex"]["path"][2],
            "2 is 2"
        )
        self.assertEqual(
            cfg["wrongDst"],
            "{commitHash1} is unchanged"
        )

    @skip_commit_slider_devtest
    def testForStaticSubstitutionRule(self):
        from utils.helpers import applySubstitutionRules
        cfg = {
                "serviceConfig": {
                    "previousKey": "previousValue"
                },
                "wrongDst": "{pathOne} is unchanged",
                "dst": {
                    "complex": {
                        "path": [
                            "{pathOne} is natural number",
                            "{pathTwo} is natural number",
                            "{pathOne} is not {pathTwo}"
                        ]
                    }
                },
                "src": {
                    "complex": {
                        "path": {
                            "one": "1",
                            "two": "2"
                        }
                    }
                }
        }
        rules = [
            {
                "name": "testRule1",
                "enabled": True,
                "type": "static",
                "placeholder": "pathOne",
                "from": "$.src.complex.path.one",
                "to": "$.dst.complex.path"
            },
            {
                "name": "testRule2",
                "enabled": True,
                "type": "static",
                "placeholder": "pathTwo",
                "from": "$.src.complex.path.two",
                "to": "$.dst.complex.path"
            }
        ]
        def applyByRef(cfg: map, rules: list, substitution: str):
            applySubstitutionRules(cfg, rules, substitution)

        applyByRef(cfg, rules, "mustBeIgnored")

        # assert substitutions
        self.assertEqual(
            cfg["dst"]["complex"]["path"][0],
            "1 is natural number"
        )

        self.assertEqual(
            cfg["dst"]["complex"]["path"][1],
            "2 is natural number"
        )
        self.assertEqual(
            cfg["dst"]["complex"]["path"][2],
            "1 is not 2"
        )
        self.assertEqual(
            cfg["wrongDst"],
            "{pathOne} is unchanged"
        )

    @skip_commit_slider_devtest
    def testForDeepUpdate(self):
        from utils.helpers import deepMapUpdate
        cfg = {
            "another": {
                "path": "not updated"
            },
            "path": {
                "to": {
                    "placeholder": "not updated"
                }
            }
        }
        cfg = deepMapUpdate(cfg, ["path", "to", "placeholder"], "updated")
        self.assertEqual(
            cfg["path"]["to"]["placeholder"],
            "updated"
        )
        self.assertEqual(
            cfg["another"]["path"],
            "not updated"
        )
