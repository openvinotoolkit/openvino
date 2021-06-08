# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import xml.etree.ElementTree as ET

def get_logger(app_name: str):
    logging.basicConfig()
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)
    return logger


def update_passrates(results: ET.SubElement):
    for device in results:
        for op in device:
            passed_tests = 0
            total_tests = 0
            for attrib in op.attrib:
                if attrib == "passrate":
                    continue
                if attrib == "passed":
                    passed_tests = int(op.attrib.get(attrib))
                total_tests += int(op.attrib.get(attrib))
            passrate = float(passed_tests * 100 / total_tests) if passed_tests < total_tests else 100
            op.set("passrate", str(round(passrate, 1)))


def update_conformance_test_counters(results: ET.SubElement, logger: logging.Logger):
    max_test_cnt = dict()
    incorrect_ops = set()
    for device in results.find("results"):
        for op in device:
            op_test_count = 0
            for attr_name in op.attrib:
                if attr_name == "passrate":
                    continue
                op_test_count += int(op.attrib.get(attr_name))
            if not op.tag in max_test_cnt.keys():
                max_test_cnt.update({op.tag: op_test_count})
            if op_test_count != max_test_cnt[op.tag]:
                incorrect_ops.add(op.tag)
                max_test_cnt[op.tag] = max(op_test_count, max_test_cnt[op.tag])

    for device in results.find("results"):
        for op in device:
            if op.tag in incorrect_ops:
                test_cnt = 0
                for attr_name in op.attrib:
                    if attr_name == "passrate":
                        continue
                    test_cnt += int(op.attrib[attr_name])
                if test_cnt != max_test_cnt[op.tag]:
                    diff = max_test_cnt[op.tag] - test_cnt
                    op.set("skipped", str(int(op.attrib["skipped"]) + diff))
                    logger.warning(f'{device.tag}: added {diff} skipped tests for {op.tag}')
    update_passrates(results)
