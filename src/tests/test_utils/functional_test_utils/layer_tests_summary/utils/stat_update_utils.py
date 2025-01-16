# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import xml.etree.ElementTree as ET

from . import conformance_utils

def update_rel_values(xml_node: ET.SubElement):
    if xml_node is None or len(xml_node.attrib) == 0:
        return
    if not "relative_all" in xml_node.attrib:
        test_cnt = int(xml_node.attrib.get("passed")) + int(xml_node.attrib.get("failed")) + int(xml_node.attrib.get("skipped")) + \
        int(xml_node.attrib.get("crashed")) + int(xml_node.attrib.get("hanged"))
        xml_node.set("relative_all", str(test_cnt))
    if not "relative_passed" in xml_node.attrib:
        xml_node.set("relative_passed", xml_node.attrib.get("passed"))

def update_passrates(results: ET.SubElement, rel_weights={}):
    for device in results:
        for op in device:
            passed_tests = 0
            total_tests = 0
            rel_passed_tests = None
            rel_all_tests_expected = None
            rel_all_tests_actual = None
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
                    if op.tag in rel_weights.keys():
                        rel_all_tests_expected = rel_weights[op.tag]
                    rel_all_tests_actual = float(op.attrib.get(attrib))
                    continue
                total_tests += int(float(op.attrib.get(attrib)))
            passrate = float(passed_tests * 100 / total_tests) if total_tests != 0 else 0
            rel_all_tests = rel_all_tests_actual if rel_all_tests_expected is None else rel_all_tests_expected
            k = 1 if rel_all_tests_expected is None else round(rel_all_tests_actual / rel_all_tests_expected)
            rel_passrate = float(rel_passed_tests * 100 / (k * rel_all_tests)) if rel_all_tests != None and rel_all_tests != 0 else 0
            op.set("passrate", f"{math.floor(passrate * 100) / 100}")
            if rel_all_tests != None and rel_passed_tests != None:
                op.set("relative_passrate", f"{math.floor(rel_passrate * 100) / 100}")



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
                    continue
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
                    if "passrate" in attr_name or attr_name == "implemented" or "relative_" in attr_name:
                        continue
                    else:
                        test_cnt += int(op.attrib[attr_name])
                if test_cnt != max_test_cnt[op.tag]:
                    diff = int(max_test_cnt[op.tag]) - test_cnt
                    op.set("skipped", str(int(op.attrib["skipped"]) + diff))
                    conformance_utils.UTILS_LOGGER.warning(f'{device.tag}: added {diff} skipped tests for {op.tag}')
            update_rel_values(op)
    update_passrates(results.find("results"))


