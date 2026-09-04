# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from collect_compile_cache_stats import parse_ccache_stats, parse_sccache_stats


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


if __name__ == "__main__":
    unittest.main()
