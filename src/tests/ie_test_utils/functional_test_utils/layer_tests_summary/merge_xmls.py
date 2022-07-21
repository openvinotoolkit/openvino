# Copyright (C) 2018-2022 Intel Corporation
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
    report_type_help = "Report type: OP or API"

    parser.add_argument("-i", "--input_folders", help=input_folders_help, nargs="*", required=True)
    parser.add_argument("-o", "--output_folder", help=output_folders_help, default=".")
    parser.add_argument("-f", "--output_filename", help=output_filename_help, default="report")
    parser.add_argument("-t", "--report_type", help=report_type_help, default="API")

    return parser.parse_args()


def aggregate_test_results(results: ET.SubElement, xml_reports: list, report_type: str):
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
                # op or api_type
                for report_entry in device_results_report:
                    if device_results.find(report_entry.tag) is not None:
                        entry = device_results.find(report_entry.tag)
                        if report_type == "OP":
                            for attr_name in device_results.find(report_entry.tag).attrib:
                                if attr_name == "passrate":
                                    continue
                                if attr_name == "implemented":
                                    xml_value = report_entry.attrib.get(attr_name) == "true"
                                    aggregated_value = entry.attrib.get(attr_name) == "true"
                                    str_value = "true" if xml_value or aggregated_value else "false"
                                    device_results.find(entry.tag).set(attr_name, str_value)
                                    continue
                                xml_value = int(report_entry.attrib.get(attr_name))
                                aggregated_value = int(entry.attrib.get(attr_name))
                                device_results.find(entry.tag).set(attr_name, str(xml_value + aggregated_value))
                        else:
                            api_results_report = xml_root.find("results").find(device.tag).find(entry.tag)
                            for api_entry in entry:
                                if api_results_report.find(api_entry.tag) is not None:
                                    for attr_name in device_results.find(report_entry.tag).find(api_entry.tag).attrib:
                                        if attr_name == "passrate":
                                            continue
                                        if attr_name == "implemented":
                                            xml_value = report_entry.find(api_entry.tag).attrib.get(attr_name) == "true"
                                            aggregated_value = api_entry.attrib.get(attr_name) == "true"
                                            str_value = "true" if xml_value or aggregated_value else "false"
                                            device_results.find(entry.tag).find(api_entry.tag).set(attr_name, str_value)
                                            continue
                                        xml_value = int(report_entry.find(api_entry.tag).attrib.get(attr_name))
                                        aggregated_value = int(api_entry.attrib.get(attr_name))
                                        device_results.find(entry.tag).find(api_entry.tag).set(attr_name, str(xml_value + aggregated_value))
                                else:
                                    entry.append(api_entry)
                    else:
                        device_results.append(report_entry)
    return timestamp


def merge_xml(input_folder_paths: list, output_folder_paths: str, output_filename: str, report_type: str):
    logger.info(f" Processing is finished")

    summary = ET.Element("report")
    results = ET.SubElement(summary, "results")
    entity_name = None
    if report_type == "OP":
        entity_name = "ops_list"
    elif report_type == "API":
        entity_name = "api_list"
    else:
        raise Exception(f"Error to create aggregated report. Incorrect report type: {report_type}")
        
    entity_list = ET.SubElement(summary, entity_name)

    for folder_path in input_folder_paths:
        if not os.path.exists(folder_path):
            logger.error(f" {folder_path} is not exist!")
            continue
        if not os.path.isdir(folder_path):
            logger.error(f" {folder_path} is not a directory!")
            continue

        xml_reports = glob.glob(os.path.join(folder_path, 'report*.xml'))

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
                ET.SubElement(entity_list, entity.tag)
        timestamp = aggregate_test_results(results, xml_reports, report_type)
        if report_type == "OP":
            utils.update_passrates(results)
        else:
            for sub_result in results:
                utils.update_passrates(sub_result)
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
    merge_xml(arguments.input_folders, arguments.output_folder, arguments.output_filename, arguments.report_type)