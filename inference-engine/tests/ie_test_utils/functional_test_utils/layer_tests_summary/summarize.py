# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from jinja2 import Environment, FileSystemLoader
import argparse
import os
from datetime import datetime

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
args = parser.parse_args()

verified_operations = [
    'Abs-0',
    'Acos-0',
    'Add-1',
    'Asin-0',
    'Assign-6',
    'AvgPool-1',
    'BatchNormInference-5',
    'BinaryConvolution-1',
    'Broadcast-1',
    'Broadcast-3',
    'Bucketize-3',
    'CTCGreedyDecoder-0',
    'CTCGreedyDecoderSeqLen-6',
    'Concat-0',
    'ConvertLike-1',
    'Convolution-1',
    'DetectionOutput-0',
    'Divide-1',
    'ExperimentalDetectronDetectionOutput-6',
    'ExperimentalDetectronGenerateProposalsSingleImage-6',
    'ExperimentalDetectronPriorGridGenerator-6',
    'ExperimentalDetectronROIFeatureExtractor-6',
    'ExperimentalDetectronTopKROIs-6',
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
    'HardSigmoid-0',
    'Interpolate-4',
    'LRN-0',
    'LSTMCell-4',
    'LSTMSequence-5',
    'LogSoftmax-5',
    'Loop-5',
    'MVN-6',
    'MaxPool-1',
    'Mish-4',
    'Multiply-1',
    'NonMaxSuppression-4',
    'NonMaxSuppression-5',
    'PSROIPooling-0',
    'Proposal-0',
    'Proposal-4',
    'RNNSequence-4',
    'ROIAlign-3',
    'ROIPooling-0',
    'Range-0',
    'Range-4',
    'ReadValue-6',
    'ReduceL1-4',
    'ReduceL2-4',
    'ReduceMean-1',
    'RegionYOLO-0',
    'Relu-0',
    'ReorgYOLO-0',
    'GRUSequence-5',
    'Round-5',
    'ShapeOf-0',
    'ShapeOf-3',
    'Sigmoid-0',
    'Sin-0',
    'SoftPlus-4',
    'Softmax-1',
    'StridedSlice-1',
    'Substract-1',
    'Swish-4',
    'Tile-0',
    'TopK-1',
    'TopK-3'
]
pass_rate_avg = dict()
general_pass_rate = dict()
general_test_count = dict()
general_passed_tests = dict()


def merge_xmls(xmls: list):
    if len(xmls) == 1:
        return xmls[0]
    summary = ET.Element("report")
    summary.set("timestamp", xmls[0].attrib["timestamp"])
    results = ET.SubElement(summary, "results")
    ops_list = ET.SubElement(summary, "ops_list")
    for xml in xmls:
        for op in xml.find("ops_list"):
            if ops_list.find(op.tag) is None:
                ET.SubElement(ops_list, op.tag)
        for device in xml.find("results"):
            device_results = results.find(device.tag)
            if device_results is None:
                results.append(device)
            else:
                for entry in device:
                    if device_results.find(entry.tag) is not None:
                        current_timestamp = datetime.strptime(xml.attrib["timestamp"], "%d-%m-%Y %H:%M:%S")
                        base_timestamp = datetime.strptime(summary.attrib["timestamp"], "%d-%m-%Y %H:%M:%S")
                        if current_timestamp > base_timestamp:
                            device_results.find(entry.tag).attrib = entry.attrib
                    else:
                        device_results.append(entry)
    return summary


xmls = []
for xml in args.xml:
    try:
        xmls.append(ET.parse(xml).getroot())
    except ET.ParseError:
        print("Error parsing", xml)

root = merge_xmls(xmls)
timestamp = root.attrib["timestamp"]
ops = []
for op in root.find("ops_list"):
    ops.append(op.tag)
ordered_ops = sorted(ops)
results = {}
for device in root.find("results"):
    results[device.tag] = {op.tag: op.attrib for op in device}
    pass_rate_avg[device.tag] = 0
    general_test_count[device.tag] = 0
    general_passed_tests[device.tag] = 0
    for op in results[device.tag]:
        pass_rate = round(float(results[device.tag][op]["passrate"]), 1)
        results[device.tag][op]["passrate"] = pass_rate
        pass_rate_avg[device.tag] += pass_rate
        general_test_count[device.tag] += (int(results[device.tag][op]["passed"]) + int(results[device.tag][op]["failed"]) +
                               int(results[device.tag][op]["crashed"]) + int(results[device.tag][op]["skipped"]))
        general_passed_tests[device.tag] += int(results[device.tag][op]["passed"])
    pass_rate_avg[device.tag] /= len(results[device.tag])
    pass_rate_avg[device.tag] = round(float(pass_rate_avg[device.tag]), 1)
    general_pass_rate[device.tag] = general_passed_tests[device.tag] * 100 / general_test_count[device.tag]
    general_pass_rate[device.tag] = round(float(general_pass_rate[device.tag]), 1)

devices = results.keys()

file_loader = FileSystemLoader('template')
env = Environment(loader=file_loader)
template = env.get_template('report_template.html')

res = template.render(ordered_ops=ordered_ops, devices=devices, results=results, timestamp=timestamp,
                      general_pass_rate=general_pass_rate, pass_rate_avg=pass_rate_avg,
                      verified_operations=verified_operations)

with open(os.path.join(args.out, "report.html"), "w") as f:
    f.write(res)
