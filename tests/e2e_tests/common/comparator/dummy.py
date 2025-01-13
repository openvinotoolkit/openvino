# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from e2e_tests.common.table_utils import make_table
from .provider import ClassProvider
import sys


class Dummy(ClassProvider):
    __action_name__ = "dummy"
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s",
        level=log.INFO,
        stream=sys.stdout)

    def __init__(self, config, infer_result, reference):
        self._config = {}
        self.infer_result = infer_result
        self.reference = reference

    def compare(self):
        log.info("Running Dummy comparator. No comparison performed")

        table_header = ["Layer Name", "Shape", "Data Range"]

        if self.infer_result:
            table_rows = []
            for layer, data in self.infer_result.items():
                table_rows.append([layer, str(data.shape), "[{:.3f}, {:.3f}]".format(data.min(), data.max())])
            log.info("Inference Engine tensors statistic:\n{}".format(make_table(table_rows, table_header)))
        if self.reference:
            table_rows = []
            for layer, data in self.reference.items():
                table_rows.append([layer, str(data.shape), "[{:.3f}, {:.3f}]".format(data.min(), data.max())])
            log.info("Reference tensors statistic:\n{}".format(make_table(table_rows, table_header)))
        self.status = True
