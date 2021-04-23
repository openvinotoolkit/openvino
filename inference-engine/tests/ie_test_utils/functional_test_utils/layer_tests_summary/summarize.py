# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import logging
import xml.etree.ElementTree as ET

from jinja2 import Environment, FileSystemLoader
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger('Summarize')
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()

    xml_help = """
        Paths to xml summary files from layer tests.
        In case of entries intersection, results will
        be merged basing on timestamp - entry from latest
        report is be kept.
    """
    out_help = "Path where to save html report"

    parser.add_argument("--xml", help=xml_help, nargs="*", required=True)
    parser.add_argument("--out", help=out_help, default="")

    return parser.parse_args()


def get_verified_op_list():
    return [
        'Abs-1',
        'Acos-1',
        'Add-1',
        'Asin-1',
        'Assign-6',
        'AvgPool-1',
        'BatchNormInference-5',
        'BinaryConvolution-1',
        'Broadcast-1',
        'Broadcast-3',
        'Bucketize-3',
        'CTCGreedyDecoder-1',
        'CTCGreedyDecoderSeqLen-6',
        'Concat-1',
        'ConvertLike-1',
        'Convolution-1',
        'DeformableConvolution-1',
        'DetectionOutput-1',
        'Divide-1',
        'ExperimentalDetectronDetectionOutput-6',
        'ExperimentalDetectronGenerateProposalsSingleImage-6',
        'ExperimentalDetectronPriorGridGenerator-6',
        'ExperimentalDetectronROIFeatureExtractor-6',
        'ExperimentalDetectronTopKROIs-6',
    'FloorMod-1'
        'GRUSequence-5',
        'Gather-1',
        'GatherElements-6',
        'GatherND-5',
        'Gelu-7',
        'GroupConvolution-1',
        'GroupConvolutionBackpropData-1',
        'GRUSequence-5',
        'HSigmoid-5',
        'HSwish-4',
        'HardSigmoid-1',
        'Interpolate-4',
        'LRN-1',
        'LSTMCell-4',
        'LSTMSequence-5',
        'LogSoftmax-5',
        'Loop-5',
        'MVN-6',
    'Maximum-1',
        'MaxPool-1',
        'Mish-4',
        'Multiply-1',
        'NonMaxSuppression-4',
        'NonMaxSuppression-5',
        'PSROIPooling-1',
        'Proposal-1',
        'Proposal-4',
        'RNNSequence-5',
        'ROIAlign-3',
        'ROIPooling-2',
        'Range-1',
        'Range-4',
        'ReadValue-6',
        'ReduceL1-4',
        'ReduceL2-4',
        'ReduceMean-1',
        'RegionYOLO-1',
        'Relu-1',
        'ReorgYOLO-2',
        'Round-5',
        'ScatterNDUpdate-4',
        'ShapeOf-1',
        'ShapeOf-3',
        'Sigmoid-1',
        'Sin-1',
        'SoftPlus-4',
        'Softmax-1',
        'Split-1',
        'StridedSlice-1',
        'Subtract-1',
        'Swish-4',
        'Tile-1',
        'TopK-1',
        'TopK-3'
    ]


def update_passrates(results: ET.SubElement):
    logger.info("Update passrates in the final report is started")
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
    logger.info("Update passrates in the final report is completed")


def merge_xmls(xml_paths: list):
    logger.info("Merging XML files is started")

    summary = ET.Element("report")
    timestamp = None
    summary_results = ET.SubElement(summary, "results")
    ops_list = ET.SubElement(summary, "ops_list")
    for xml_path in xml_paths:
        try:
            xml_root = ET.parse(xml_path).getroot()
            logger.info(f'Info from {xml_path} is adding to the final summary')
        except ET.ParseError:
            logger.error(f'Error parsing {xml_path}')

        if timestamp is None or timestamp < xml_root.attrib["timestamp"]:
            logger.info(f'Timestamp is updated from {timestamp} to {xml_root.attrib["timestamp"]}')
            timestamp = xml_root.attrib["timestamp"]

        for op in xml_root.find("ops_list"):
            if ops_list.find(op.tag) is None:
                ET.SubElement(ops_list, op.tag)

        for device in xml_root.find("results"):
            device_results = summary_results.find(device.tag)
            if device_results is None:
                summary_results.append(device)
            else:
                for op_result in device:
                    current_op_res = device_results.find(op_result.tag)
                    if current_op_res is not None:
                        # workaround for unsaved reports
                        total_tests_count_xml, total_tests_count_summary = (0, 0)
                        for attr_name in device_results.find(op_result.tag).attrib:
                            if attr_name == "passrate":
                                continue
                            total_tests_count_xml += int(op_result.attrib.get(attr_name))
                            total_tests_count_summary += int(current_op_res.attrib.get(attr_name))
                        if total_tests_count_xml > total_tests_count_xml:
                            logger.warning(f'Test counter is different in {op_result.tag} for {device.tag}'\
                                           f'({total_tests_count_xml} vs {total_tests_count_xml})')
                            for attr_name in device_results.find(op_result.tag).attrib:
                                if attr_name == "passrate":
                                    continue
                                xml_value = int(op_result.attrib.get(attr_name))
                                device_results.find(current_op_res.tag).set(attr_name, str(xml_value))
                    else:
                        device_results.append(op_result)
    update_passrates(summary_results)
    summary.set("timestamp", timestamp)
    logger.info("Merging XML files is competed")
    return summary


def collect_statistic(root: ET.Element):
    logger.info("Statistic collecting is started")
    trusted_ops = dict()
    pass_rate_avg = dict()
    general_pass_rate = dict()
    general_test_count = dict()
    general_passed_tests = dict()

    results = dict()
    for device in root.find("results"):
        results[device.tag] = {op.tag: op.attrib for op in device}

        pass_rate_avg[device.tag] = 0
        general_test_count[device.tag] = 0
        general_passed_tests[device.tag] = 0
        trusted_ops[device.tag] = 0
        for op in results[device.tag]:
            pass_rate = round(float(results[device.tag][op]["passrate"]), 1)
            results[device.tag][op]["passrate"] = pass_rate

            pass_rate_avg[device.tag] += pass_rate
            if pass_rate == 100.:
                trusted_ops[device.tag] += 1
            general_test_count[device.tag] += (
                    int(results[device.tag][op]["passed"]) + int(results[device.tag][op]["failed"]) +
                    int(results[device.tag][op]["crashed"]) + int(results[device.tag][op]["skipped"]))
            general_passed_tests[device.tag] += int(results[device.tag][op]["passed"])

        pass_rate_avg[device.tag] /= len(results[device.tag])
        pass_rate_avg[device.tag] = round(float(pass_rate_avg[device.tag]), 1)
        general_pass_rate[device.tag] = general_passed_tests[device.tag] * 100 / general_test_count[device.tag]
        general_pass_rate[device.tag] = round(float(general_pass_rate[device.tag]), 1)

    devices = results.keys()
    logger.info("Statistic collecting is completed")
    return devices, results, general_pass_rate, pass_rate_avg, general_test_count, trusted_ops


def create_summary(summary_root: ET.Element, output_folder: str):
    device_list, results, general_pass_rate, pass_rate_avg, general_test_count, trusted_ops = \
        collect_statistic(summary_root)

    timestamp = summary_root.attrib["timestamp"]

    op_list = list()
    for op in summary_root.find("ops_list"):
        op_list.append(op.tag)
    op_list = sorted(op_list)

    file_loader = FileSystemLoader('template')
    env = Environment(loader=file_loader)
    template = env.get_template('report_template.html')

    verified_operations = get_verified_op_list()

    res_summary = template.render(ordered_ops=op_list, devices=device_list, results=results, timestamp=timestamp,
                                  general_pass_rate=general_pass_rate, pass_rate_avg=pass_rate_avg,
                                  verified_operations=verified_operations, trusted_ops=trusted_ops,
                                  general_test_count=general_test_count)

    report_path = os.path.join(output_folder, "report.html")
    with open(report_path, "w") as f:
        logger.info(f'Final report is saved to {report_path}')
        f.write(res_summary)


if __name__ == "__main__":
    args = parse_arguments()
    summary_root = merge_xmls(args.xml)
    create_summary(summary_root, args.out)
