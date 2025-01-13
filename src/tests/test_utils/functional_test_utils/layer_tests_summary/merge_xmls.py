# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import glob

import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib

from utils.conformance_utils import get_logger
from utils import stat_update_utils
from utils.constants import OP_CONFORMANCE, API_CONFORMANCE

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element
SubElement = ET_defused.SubElement

logger = get_logger('xml_merge_tool')

def parse_arguments():
    parser = argparse.ArgumentParser()

    input_folders_help = "Paths to folders with reports to merge"
    output_folders_help = "Path to folder to save report"
    output_filename_help = "Output report filename"
    report_type_help = "Report type: OP or API"
    merge_device_id_help = "Merge all devices with suffix to one main device. Example: GPU.0 and GPU.1 -> GPU"

    parser.add_argument("-i", "--input_folders", help=input_folders_help, nargs="*", required=True)
    parser.add_argument("-o", "--output_folder", help=output_folders_help, default=".")
    parser.add_argument("-f", "--output_filename", help=output_filename_help, default="report")
    parser.add_argument("-t", "--report_type", help=report_type_help, default="OP")
    parser.add_argument("-m", "--merge_device_id", help=merge_device_id_help, default=False)

    return parser.parse_args()


def update_result_node(xml_node: SubElement, aggregated_res: SubElement):
    stat_update_utils.update_rel_values(aggregated_res)
    stat_update_utils.update_rel_values(xml_node)
    for attr_name in xml_node.attrib:
        if attr_name == "passrate" or attr_name == "relative_passrate":
            continue
        if attr_name == "implemented":
            xml_value = xml_node.attrib.get(attr_name) == "true"
            aggregated_value = aggregated_res.attrib.get(attr_name) == "true"
            str_value = "true" if xml_value or aggregated_value else "false"
            aggregated_res.set(attr_name, str_value)
            continue
        xml_value = float(xml_node.attrib.get(attr_name)) if "relative_" in attr_name else int(xml_node.attrib.get(attr_name))
        aggregated_value = float(aggregated_res.attrib.get(attr_name)) if "relative_" in attr_name else int(aggregated_res.attrib.get(attr_name))
        aggregated_res.set(attr_name, str(xml_value + aggregated_value))


def aggregate_test_results(aggregated_results: SubElement, xml_reports: list,
                           report_type: str, merge_device_suffix=False):
    aggregated_timestamp = None
    for xml in xml_reports:
        # logger.info(f" Processing: {xml}")
        try:
            xml_root = ET.parse(xml).getroot()
        except ET.ParseError:
            # logger.error(f' {xml} is corrupted and skipped')
            continue
        xml_results = xml_root.find("results")
        xml_timestamp = xml_root.get("timestamp")
        if aggregated_timestamp is None or xml_timestamp > aggregated_timestamp:
            aggregated_timestamp = xml_timestamp
        for xml_device_entry in xml_results:
            if merge_device_suffix and "." in xml_device_entry.tag:
                device_name = xml_device_entry.tag[:xml_device_entry.tag.find("."):]
                new_data = ET.tostring(xml_device_entry).decode('utf8').replace(xml_device_entry.tag, device_name)
                xml_device_entry = ET.fromstring(new_data)
            device_name = xml_device_entry.tag
            aggregated_device_results = aggregated_results.find(device_name)
            # example: ov_plugin or Add-1
            for xml_results_entry in xml_device_entry:
                if report_type == OP_CONFORMANCE or report_type == OP_CONFORMANCE.lower():
                    aggregated_results_entry = None
                    if not aggregated_device_results is None:
                        aggregated_results_entry = aggregated_device_results.find(xml_results_entry.tag)
                    if aggregated_results_entry is None:
                        stat_update_utils.update_rel_values(xml_results_entry)
                        if aggregated_device_results is None:
                            aggregated_results.append(xml_device_entry)
                            aggregated_device_results = aggregated_results.find(device_name)
                            break
                        else:
                            aggregated_device_results.append(xml_results_entry)
                        continue
                    update_result_node(xml_results_entry, aggregated_results_entry)
                else:
                    aggregated_results_entry = None
                    if aggregated_device_results is None:
                        aggregated_results.append(xml_device_entry)
                        break
                    else:
                        aggregated_results_entry = aggregated_device_results.find(xml_results_entry.tag)
                    if aggregated_results_entry:
                        for xml_real_device_entry in xml_results_entry:
                            aggregated_real_device_api_report = None
                            aggregated_real_device_api_report = aggregated_results_entry.find(xml_real_device_entry.tag)
                            if aggregated_real_device_api_report is None:
                                stat_update_utils.update_rel_values(xml_results_entry)
                                aggregated_results_entry.append(xml_real_device_entry)
                                continue
                            update_result_node(xml_real_device_entry, aggregated_real_device_api_report)
                    else:
                        aggregated_device_results.append(xml_results_entry)

    return aggregated_timestamp


def merge_xml(input_folder_paths: list, output_folder_paths: str, output_filename: str,
              report_type: str, merge_device_suffix=False):
    logger.info(f" Processing is finished")

    summary = Element("report")
    results = SubElement(summary, "results")
    entity_name = None
    if report_type == OP_CONFORMANCE.lower() or report_type == OP_CONFORMANCE:
        entity_name = "ops_list"
    elif report_type == API_CONFORMANCE.lower() or report_type == API_CONFORMANCE:
        entity_name = "api_list"
    else:
        raise Exception(f"Error to create aggregated report. Incorrect report type: {report_type}")
    
    entity_list = SubElement(summary, entity_name)

    for folder_path in input_folder_paths:
        if not os.path.exists(folder_path):
            logger.error(f" {folder_path} is not exist!")
            continue
        if not os.path.isdir(folder_path):
            logger.error(f" {folder_path} is not a directory!")
            continue

        xml_reports = None
        if report_type == OP_CONFORMANCE.lower() or report_type == OP_CONFORMANCE:
            xml_reports = glob.glob(os.path.join(folder_path, 'report_op*.xml'))
        elif report_type == API_CONFORMANCE.lower() or report_type == API_CONFORMANCE:
            xml_reports = glob.glob(os.path.join(folder_path, 'report_api*.xml'))
        logger.info(f"Num of XML: {len(xml_reports)}")

        xml_root = None
        for xml_report in xml_reports:
            try:
                xml_root = ET.parse(xml_report).getroot()
                break
            except ET.ParseError:
                logger.error(f'{xml_report} is incorrect! Error to get a xml root')
        if xml_root is None:
            logger.error(f'{folder_path} does not contain the correct xml files')
        for entity in xml_root.find(entity_name):
            if entity_list.find(entity.tag) is None:
                entity_node = SubElement(entity_list, entity.tag)
                for op_attrib in entity.attrib:
                    entity_node.set(op_attrib, entity.get(op_attrib))
        timestamp = aggregate_test_results(results, xml_reports, report_type, merge_device_suffix)
        if report_type == "OP":
            stat_update_utils.update_passrates(results)
        else:
            for sub_result in results:
                stat_update_utils.update_passrates(sub_result)
        summary.set("timestamp", timestamp)
        logger.info(f" Processing is finished")

        if not os.path.exists(output_folder_paths):
            os.mkdir(output_folder_paths)
        out_file_path = os.path.join(output_folder_paths, f'{output_filename}.xml')
        with open(out_file_path, "w") as xml_file:
            xml_file.write(ET.tostring(summary).decode('utf8'))
            logger.info(f" Final report is saved to file: '{out_file_path}'")
    if xml_root is None:
        raise Exception("Error to make a XML root. Exit the app")


if __name__ == "__main__":
    arguments = parse_arguments()
    merge_xml(arguments.input_folders, arguments.output_folder, arguments.output_filename, arguments.report_type, arguments.merge_device_id)
