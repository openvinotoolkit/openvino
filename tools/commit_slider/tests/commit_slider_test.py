# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest import TestCase
from tests import skip_commit_slider_devtest

sys.path.append('./')
from test_util import getExpectedCommit, \
    getBordersByTestData, getActualCommit, getCSOutput, \
    createRepoAndUpdateData, runCSAndCheckPattern
from utils.break_validator import validateBMOutput, BmValidationError
from test_data import FirstBadVersionData, FirstValidVersionData,\
    BmStableData, BmValidatorSteppedBreakData, BmValidatorSteppedBreakData2,\
    BenchmarkAppDataUnstable, BenchmarkAppDataStable, BenchmarkAppNoDegradationData,\
    BenchmarkAppUnstableDevData, BenchmarkAppWrongPathData, BenchmarkAppPathFoundData,\
    BenchmarkFirstFixedAppData, AcModeData, BenchmarkMetricData, CustomizedLogData, \
    MultiConfigData, ConfigMultiplicatorData, ConfigMultiplicatorWithKeyData, \
    AcModeDataBitwise, CompareBlobsData, CompareBlobsMulOutputData, CompareBlobsAutomatchData, \
    BrokenCompilationData, TemplateData, CrossCheckBadAppl, CrossCheckBadModel, CrossCheckPerformance, \
    CrossCheckPerformanceSeparateMode, CrossCheckPerformanceSeparateTemplate, CrossCheckPerformanceSeparateTemplateBadModel, \
    TableTemplate

class CommitSliderTest(TestCase):
    @skip_commit_slider_devtest
    def testFirstValidVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstValidVersionData())
        actualCommit, _ = getActualCommit(updatedData)

        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBrokenCompilation(self):
        breakCommit, updatedData = getExpectedCommit(
            BrokenCompilationData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBrokenCompTmplate(self):
        breakCommit, updatedData = getExpectedCommit(
            TemplateData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testFirstBadVersion(self):
        breakCommit, updatedData = getExpectedCommit(
            FirstBadVersionData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testCrossCheckBadAppl(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckBadAppl())
        res = runCSAndCheckPattern(updatedData, ["failed", "failed", "success", "success"])

        self.assertEqual(True, res)

    @skip_commit_slider_devtest
    def testCrossCheckBadModel(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckBadModel())
        res = runCSAndCheckPattern(updatedData, ["success", "failed", "success", "failed"])

        self.assertEqual(True, res)

    @skip_commit_slider_devtest
    def testCrossCheckPerformance(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckPerformance())
        res = runCSAndCheckPattern(updatedData, ["500.0 FPS", "500.0 FPS", "1000.0 FPS", "1000.0 FPS"])

        self.assertEqual(True, res)

    @skip_commit_slider_devtest
    def testCrossCheckPerformanceSeparateMode(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckPerformanceSeparateMode())
        res = runCSAndCheckPattern(updatedData, ["500.0", "500.0", "1000.0", "1000.0"])

        self.assertEqual(True, res)

    @skip_commit_slider_devtest
    def testCrossCheckPerformanceSeparateTemplate(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckPerformanceSeparateTemplate())
        res = runCSAndCheckPattern(updatedData, ["rootcause", "OV"])

        self.assertTrue(res)

    @skip_commit_slider_devtest
    def testTableTemplate(self):
        updatedData = createRepoAndUpdateData(
            TableTemplate())
        res = runCSAndCheckPattern(updatedData, ["rootcause", "OV"])

        self.assertTrue(res)

    @skip_commit_slider_devtest
    def testCrossCheckPerformanceSeparateTemplateBadModel(self):
        updatedData = createRepoAndUpdateData(
            CrossCheckPerformanceSeparateTemplateBadModel())
        res = runCSAndCheckPattern(updatedData, ["rootcause", "Model"])

        self.assertTrue(res)

    @skip_commit_slider_devtest
    def testCustomizedLog(self):
        breakCommit, updatedData = getExpectedCommit(
            CustomizedLogData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testMultiConfig(self):
        _, updatedData = getExpectedCommit(
            MultiConfigData())

        self.assertEqual(
            getCSOutput(updatedData),
            "\n\n".join(['cfg #{n}'.format(n=n) for n in range(3)]) + "\n")

    @skip_commit_slider_devtest
    def testConfigMultiplicatorByKey(self):
        from utils.helpers import multiplyCfgByKey
        from utils.helpers import deepCopyJSON
        testData = ConfigMultiplicatorData()
        self.assertEqual(
            multiplyCfgByKey(testData.testCfg),
            deepCopyJSON(testData.multipliedCfg))

    @skip_commit_slider_devtest
    def testRunCSMultiplicatedCfgByKey(self):
        _, updatedData = getExpectedCommit(
            ConfigMultiplicatorWithKeyData())

        self.assertEqual(
            getCSOutput(updatedData),
            "\n\n".join(['cfg #{n}'.format(n=n) for n in range(3)]) + "\n")

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
    def testACModeBitwise(self):
        breakCommit, updatedData = getExpectedCommit(
            AcModeDataBitwise())
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
        deepMapUpdate(cfg, ["path", "to", "placeholder"], "updated")
        self.assertEqual(
            cfg["path"]["to"]["placeholder"],
            "updated"
        )
        self.assertEqual(
            cfg["another"]["path"],
            "not updated"
        )

    @skip_commit_slider_devtest
    def testAccuracyBase(self):
        breakCommit, updatedData = getExpectedCommit(
            CompareBlobsData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testAccuracyMultipleOutput(self):
        breakCommit, updatedData = getExpectedCommit(
            CompareBlobsMulOutputData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testAccuracyMultipleOutputAutomatch(self):
        breakCommit, updatedData = getExpectedCommit(
            CompareBlobsAutomatchData())
        actualCommit, _ = getActualCommit(updatedData)
        self.assertEqual(breakCommit, actualCommit)

    @skip_commit_slider_devtest
    def testBlobMatching(self):
        from utils.helpers import getBlobMatch
        fileList1 = [
            "#001_Some_node_type_1.ieb",
            "#002_Some_node_type_2.ieb",
            "#003_Some_node_type_3.ieb",
            "#004_Some_node_type_4.ieb",
            "#005_Some_node_type_5.ieb"]
        fileList2 = [
            "#006_Some_changed_node_type_1.ieb",
            "#007_Some_changed_node_type_3.ieb",
            "#008_Some_changed_node_type_2.ieb",
            "#008_Some_changed_node_type_5.ieb",
            "#008_Some_changed_node_type_4.ieb"
        ]
        expectedMatch = [
            tuple([0, 0]), tuple([1, 2]), tuple([2, 1]), tuple([3, 4]), tuple([4, 3])]
        curMatch = getBlobMatch(fileList1, fileList2)
        actualMatch = [tuple([fileList1.index(v[0]), fileList2.index(v[1])]) for v in curMatch]
        diff = set(expectedMatch).difference(set(actualMatch))
        self.assertEqual(diff, set())
