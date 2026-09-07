# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import unittest.mock

from collect_compile_cache_stats import (
    extract_summary_metrics,
    format_step_summary_markdown,
    parse_ccache_stats,
    parse_sccache_stats,
)


SCCACHE_SAMPLE = """\
Compile requests                  10886
Compile requests executed         10884
Cache hits                        10793
Cache hits (C/C++)                10793
Cache misses                         70
Cache misses (C/C++)                 70
Cache timeouts                        0
Cache read errors                     0
Forced recaches                       0
Cache write errors                    0
Compilation failures                 10
Cache errors                         11
Cache errors (C/C++)                 11
Non-cacheable compilations            0
Non-cacheable calls                   2
Non-compilation calls                 0
Unsupported compiler calls            0
Average cache write               0.080 s
Average compiler                  2.805 s
Average cache read hit            0.015 s
Failed distributed compilations       0

Non-cacheable reasons:
multiple input files                  2

Cache location                  azblob, name: sccache, prefix: /android_x64/
Version (client)                0.7.5
"""

CCACHE_SAMPLE = """\
Cacheable calls:   9758 / 9793 (99.64%)
  Hits:            8754 / 9758 (89.71%)
    Direct:        7857 / 8754 (89.75%)
    Preprocessed:   897 / 8754 (10.25%)
  Misses:          1004 / 9758 (10.29%)
Uncacheable calls:   35 / 9793 ( 0.36%)
Local storage:
  Cache size (GB):  3.0 /  3.0 (99.97%)
  Cleanups:         113
  Hits:            8754 / 9758 (89.71%)
  Misses:          1004 / 9758 (10.29%)
"""


class TestParseSccacheStats(unittest.TestCase):
    def test_parses_sample_and_computes_hit_rate(self):
        report = parse_sccache_stats(SCCACHE_SAMPLE)
        self.assertEqual(report["tool"], "sccache")
        self.assertEqual(report["metrics"]["compile_requests"], 10886)
        self.assertEqual(report["metrics"]["cache_hits"], 10793)
        self.assertEqual(report["metrics"]["cache_misses"], 70)
        self.assertAlmostEqual(report["computed"]["cache_hit_rate"], 10793 / (10793 + 70), places=6)
        self.assertAlmostEqual(report["computed"]["cache_hit_percentage"], 99.3556, places=3)
        self.assertEqual(report["non_cacheable_reasons"]["multiple_input_files"], 2)
        self.assertIn("azblob", report["metadata"]["cache_location"])
        self.assertEqual(report["metadata"]["version_client"], "0.7.5")
        self.assertAlmostEqual(report["metrics"]["average_cache_write"], 0.080)


class TestParseCcacheStats(unittest.TestCase):
    def test_parses_sample_and_computes_nested_rates(self):
        report = parse_ccache_stats(CCACHE_SAMPLE)
        self.assertEqual(report["tool"], "ccache")
        self.assertEqual(report["metrics"]["cacheable_calls"]["numerator"], 9758)
        self.assertEqual(report["metrics"]["hits"]["numerator"], 8754)
        self.assertEqual(report["metrics"]["misses"]["numerator"], 1004)
        self.assertEqual(report["metrics"]["preprocessed"]["numerator"], 897)
        self.assertEqual(report["metrics"]["direct"]["numerator"], 7857)
        self.assertEqual(report["local_storage"]["cleanups"], 113)
        self.assertAlmostEqual(
            report["computed"]["cache_hit_rate_of_cacheable"],
            8754 / 9758,
            places=6,
        )
        self.assertAlmostEqual(
            report["computed"]["direct_hit_rate_of_hits"],
            7857 / 8754,
            places=6,
        )

    def test_parses_ccache_lines_with_spaced_percent_parentheses(self):
        report = parse_ccache_stats(CCACHE_WINDOWS_CC_SAMPLE)
        self.assertEqual(report["metrics"]["misses"]["numerator"], 258)
        self.assertEqual(report["metrics"]["preprocessed"]["numerator"], 3)
        self.assertEqual(report["metrics"]["uncacheable_calls"]["numerator"], 35)
        self.assertEqual(report["local_storage"]["misses"]["numerator"], 258)
        metrics = extract_summary_metrics(report)
        self.assertEqual(metrics["cache_hits"], 4054)
        self.assertEqual(metrics["cache_misses"], 258)
        self.assertEqual(metrics["cache_hit_rate"], "94.02%")
        markdown = format_step_summary_markdown({**report, "build_label": "cc-collect-openvino-main"})
        self.assertIn("| 4054 | 258 | 94.02% | — |", markdown)


CCACHE_WINDOWS_CC_SAMPLE = """\
Cacheable calls:   4312 / 4347 (99.19%)
  Hits:            4054 / 4312 (94.02%)
    Direct:        4051 / 4054 (99.93%)
    Preprocessed:     3 / 4054 ( 0.07%)
  Misses:           258 / 4312 ( 5.98%)
Uncacheable calls:   35 / 4347 ( 0.81%)
Local storage:
  Cache size (GB):  1.3 /  3.0 (44.65%)
  Hits:            4054 / 4312 (94.02%)
  Misses:           258 / 4312 ( 5.98%)
"""


class TestStepSummaryMarkdown(unittest.TestCase):
    def test_sccache_summary_table(self):
        report = parse_sccache_stats(SCCACHE_SAMPLE)
        report["build_label"] = "openvino-main"
        markdown = format_step_summary_markdown(report)
        self.assertIn("## Compile cache statistics (openvino main)", markdown)
        self.assertIn("| 10793 | 70 | 99.36% | 11 |", markdown)

    def test_explicit_summary_title(self):
        report = parse_sccache_stats(SCCACHE_SAMPLE)
        with unittest.mock.patch.dict(
            "os.environ",
            {"COMPILE_CACHE_SUMMARY_TITLE": "Compile cache statistics (openvino main build)"},
            clear=False,
        ):
            markdown = format_step_summary_markdown(report)
        self.assertIn("## Compile cache statistics (openvino main build)", markdown)

    def test_job_title_when_no_build_label(self):
        report = parse_sccache_stats(SCCACHE_SAMPLE)
        with unittest.mock.patch.dict(
            "os.environ",
            {"GITHUB_JOB": "build_linux", "COMPILE_CACHE_SUMMARY_TITLE": ""},
            clear=False,
        ):
            markdown = format_step_summary_markdown(report)
        self.assertIn("## build_linux", markdown)

    def test_ccache_summary_metrics(self):
        report = parse_ccache_stats(CCACHE_SAMPLE)
        metrics = extract_summary_metrics(report)
        self.assertEqual(metrics["cache_hits"], 8754)
        self.assertEqual(metrics["cache_misses"], 1004)
        self.assertEqual(metrics["cache_hit_rate"], "89.71%")
        self.assertIsNone(metrics["errors"])
        self.assertEqual(metrics["cache_size_gb"], "3 GB")
        self.assertEqual(metrics["cache_max_size_gb"], "3 GB")
        self.assertEqual(metrics["cache_saturation"], "99.97%")

    def test_ccache_summary_table_includes_cache_size_columns(self):
        report = parse_ccache_stats(CCACHE_SAMPLE)
        markdown = format_step_summary_markdown(report)
        self.assertIn("| Cache size | Cache max size | Cache saturation |", markdown)
        self.assertIn("| 3 GB | 3 GB | 99.97% |", markdown)

    def test_sccache_summary_table_omits_cache_size_columns(self):
        report = parse_sccache_stats(SCCACHE_SAMPLE)
        markdown = format_step_summary_markdown(report)
        self.assertNotIn("Cache size", markdown)
        self.assertNotIn("Cache saturation", markdown)


if __name__ == "__main__":
    unittest.main()
