# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import glob

import xml.etree.ElementTree as ET

from utils import utils

logger = utils.get_logger('XmlMerger')

def parse_arguments():
    parser = argparse.ArgumentParser()

    input_folders_help = "Paths to folders with reports to merge"
    output_folders_help = "Path to folder to save report"
    output_filename_help = "Output report filename"

    parser.add_argument("-i", "--input_folders", help=input_folders_help, nargs="*", required=True)
    parser.add_argument("-o", "--output_folder", help=output_folders_help, default="")
    parser.add_argument("-f", "--output_filename", help=output_filename_help, default="report")

    return parser.parse_args()


def aggregate_test_results(results: ET.SubElement, xml_reports: list):
    timestamp = None
    for xml in xml_reports:
        logger.info(f" Processing: {xml}")
        try:
            xml_root = ET.parse(xml).getroot()
        except ET.ParseError:
            logger.error(f' {xml} is corrupted and skipped')
            continue
        xml_timestamp = xml_root.get("timestamp")
        if (timestamp is None) or (xml_timestamp < timestamp):
            timestamp = xml_timestamp
        for device in xml_root.find("results"):
            device_results = results.find(device.tag)
            if device_results is None:
                results.append(device)
            else:
                device_results_report = xml_root.find("results").find(device.tag)
                for op in device_results_report:
                    if device_results.find(op.tag) is not None:
                        entry = device_results.find(op.tag)
                        for attr_name in device_results.find(op.tag).attrib:
                            if attr_name == "passrate":
                                continue
                            xml_value = int(op.attrib.get(attr_name))
                            aggregated_value = int(entry.attrib.get(attr_name))
                            device_results.find(entry.tag).set(attr_name, str(xml_value + aggregated_value))
                    else:
                        device_results.append(op)
    return timestamp


def merge_xml(input_folder_paths: list, output_folder_paths: str, output_filename: str):
    logger.info(f" Processing is finished")

    summary = ET.Element("report")
    results = ET.SubElement(summary, "results")
    ops_list = ET.SubElement(summary, "ops_list")

    for folder_path in input_folder_paths:
        if not os.path.exists(folder_path):
            logger.error(f" {folder_path} is not exist!")
            continue
        if not os.path.isdir(folder_path):
            logger.error(f" {folder_path} is not a directory!")
            continue

        xml_reports = glob.glob(os.path.join(folder_path, 'report*.xml'))

        xml_root = ET.parse(xml_reports[0]).getroot()
        for op in xml_root.find("ops_list"):
            if ops_list.find(op.tag) is None:
                ET.SubElement(ops_list, op.tag)

        timestamp = aggregate_test_results(results, xml_reports)
        utils.update_passrates(results)
        summary.set("timestamp", timestamp)
        logger.info(f" Processing is finished")

        if not os.path.exists(output_folder_paths):
            os.mkdir(output_folder_paths)
        out_file_path = os.path.join(output_folder_paths, f'{output_filename}.xml')
        with open(out_file_path, "w") as xml_file:
            xml_file.write(ET.tostring(summary).decode('utf8'))
            logger.info(f" Final report is saved to file: '{out_file_path}'")


if __name__ == "__main__":
    arguments = parse_arguments()
    merge_xml(arguments.input_folders, arguments.output_folder, arguments.output_filename)
