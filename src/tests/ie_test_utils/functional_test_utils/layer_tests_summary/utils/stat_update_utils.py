# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET

from . import conformance_utils

def update_passrates(results: ET.SubElement):
    for device in results:
        for op in device:
            passed_tests = 0
            total_tests = 0
            rel_passed_tests = 0
            rel_all_tests = 0
            for attrib in op.attrib:
                if attrib == "passrate" or attrib == "relative_passrate":
                    continue
                if attrib == "implemented":
                    continue
                elif attrib == "passed":
                    passed_tests = int(op.attrib.get(attrib))
                elif attrib == "relative_passed":
                    rel_passed_tests = float(op.attrib.get(attrib))
                    continue
                elif attrib == "relative_all":
                    rel_all_tests = float(op.attrib.get(attrib))
                    continue
                total_tests += int(op.attrib.get(attrib))
            passrate = float(passed_tests * 100 / total_tests) if total_tests != 0 else 0
            rel_passrate = float(rel_passed_tests * 100 / rel_all_tests) if rel_all_tests != 0 else 0
            op.set("passrate", "%.2f"%passrate)
            op.set("relative_passrate", "%.2f"%rel_passrate)


def update_conformance_test_counters(results: ET.SubElement):
    max_test_cnt = dict()
    incorrect_ops = set()
    for device in results.find("results"):
        for op in device:
            op_test_count = 0
            for attr_name in op.attrib:
                if attr_name == "passrate" or attr_name == "implemented" or attr_name == "relative_passrate":
                    continue
                elif "relative_" in attr_name:
                    op_test_count += float(op.attrib.get(attr_name))
                else:
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
                    if attr_name == "passrate" or attr_name == "implemented" or attr_name == "relative_passrate":
                        continue
                    elif "relative_" in attr_name:
                        test_cnt += float(op.attrib[attr_name])
                    else:
                        test_cnt += int(op.attrib[attr_name])
                if test_cnt != max_test_cnt[op.tag]:
                    diff = max_test_cnt[op.tag] - test_cnt
                    op.set("skipped", str(int(op.attrib["skipped"]) + diff))
                    conformance_utils.UTILS_LOGGER.warning(f'{device.tag}: added {diff} skipped tests for {op.tag}')
    update_passrates(results)

