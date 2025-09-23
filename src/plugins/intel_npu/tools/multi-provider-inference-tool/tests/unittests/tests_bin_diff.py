#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import json
import os
import pathlib
import shutil
import sys
import unittest

sys.path.append("../..")

from array import array

from common.provider_description import TensorsInfoPrinter
from pathlib import Path
from tools.bin_diff import compare_blobs

class UtilsTests_Tools_bin_diff(unittest.TestCase):

    def setUp(self):
        self.sandbox_dir = "UtilsTests_Tools_bin_diff"
        self.temporary_directories = []
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def tearDown(self):
        for d in self.temporary_directories:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def get_tmp_blob_file_path(self, file_name):
        return Path(self.sandbox_dir) / file_name

    def test_compate_similar_blobs(self):
        integer_array = array('i', [0, -1, 2, -3, 4, -5, 6, -7, 8, -9])
        lhs_blob_path = self.get_tmp_blob_file_path("test_compate_similar_blobs_lhs.blob")
        with open(lhs_blob_path, "wb") as file:
            integer_array.tofile(file)

        rhs_blob_path = self.get_tmp_blob_file_path("test_compate_similar_blobs_rhs.blob")
        with open(rhs_blob_path, "wb") as file:
            integer_array.tofile(file)

        integer_types= ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32"]
        nrmse_results = set()
        for t in integer_types:
            nrmse_results.add(compare_blobs(lhs_blob_path, rhs_blob_path, t))

        # all integer comparison must give the same nrmse
        self.assertEqual(len(nrmse_results), 1)
        self.assertAlmostEqual(len(nrmse_results), 1, places=4)

    def test_compate_different_blobs(self):
        lhs_blob_array = None
        rhs_blob_array = None
        if sys.byteorder == "little":
            lhs_blob_array = array('b', [0, 0, -1, 0, 2, 0, -3, 0, 4, 0, -5, 0, 6, 0, -7, 0, 8, 0, -9, 0])
            rhs_blob_array = array('b', [-9, 0, 8, 0, -7, 0, 6, 0, -5, 0, 4, 0, -3, 0, 2, 0, -1, 0, 0, 0])
        else:
            lhs_blob_array = array('b', [0, 0, 0, -1, 0, 2, 0, -3, 0, 4, 0, -5, 0, 6, 0, -7, 0, 8, 0, -9])
            rhs_blob_array = array('b', [0, -9, 0, 8, 0, -7, 0, 6, 0, -5, 0, 4, 0, -3, 0, 2, 0, -1, 0, 0])

        lhs_blob_path = self.get_tmp_blob_file_path("test_compate_different_blobs_lhs.blob")
        with open(lhs_blob_path, "wb") as file:
            lhs_blob_array.tofile(file)

        rhs_blob_path = self.get_tmp_blob_file_path("test_compate_different_blobs_rhs.blob")
        with open(rhs_blob_path, "wb") as file:
            rhs_blob_array.tofile(file)
        integer_types= ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32"]
        nrmse_results = {}
        for t in integer_types:
            nrmse_results[t] = compare_blobs(lhs_blob_path, rhs_blob_path, t)

        # data in initial blobs was chosen thoroughly, so that these condition must be true
        self.assertTrue(nrmse_results["int8"] > nrmse_results["uint8"])
        self.assertEqual(nrmse_results["int16"], nrmse_results["uint16"])
        self.assertEqual(nrmse_results["int32"], nrmse_results["uint32"])
        self.assertEqual(nrmse_results["int64"], nrmse_results["uint64"])
        self.assertNotEqual(nrmse_results["float16"], nrmse_results["float32"])

if __name__ == '__main__':
    unittest.main()
