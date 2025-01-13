# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import csv
from pathlib import Path
import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib

from jinja2 import Environment, FileSystemLoader

from utils.conformance_utils import get_logger
from utils import stat_update_utils
from utils.constants import REL_WEIGHTS_FILENAME

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element
SubElement = ET_defused.SubElement

NOT_RUN = "NOT RUN"
NA = "N/A"

STATUS_CSV_ORDER = ["implemented", "passed", "failed", "skipped", "crashed", "hanged", "passrate", "relative_passrate"]

logger = get_logger('conformance_summary')


def parse_arguments():
    parser = argparse.ArgumentParser()

    xml_help = """
        Paths to xml summary files from layer tests.
        In case of entries intersection, results will
        be merged basing on timestamp - entry from latest
        report is be kept.
    """
    out_help = "Path where to save html report"
    report_tag = "Report tag"
    report_version = "Report version"
    output_filename_help = "Output report filename"
    conformance_mode_help = "Allow to align test number"
    csv_help = "Allow to serialize report as csv file"
    expected_devices_help = "List of expected devices"
    rel_weights_help = "Path to dir/file with rel weights"
    report_type_help = "Report type: OP or API"

    parser.add_argument("--xml", help=xml_help, nargs="*", required=True)
    parser.add_argument("--out", help=out_help, default="")
    parser.add_argument("--output_filename", help=output_filename_help, default="report")
    parser.add_argument("--report_tag", help=report_tag, default="")
    parser.add_argument("--report_version", help=report_version, default="")
    parser.add_argument("--conformance_mode", help=conformance_mode_help, default=False)
    parser.add_argument("--csv", help=csv_help, default=False)
    parser.add_argument("--expected_devices", help=expected_devices_help, nargs="*", required=False)
    parser.add_argument("--rel_weights", help=rel_weights_help, type=str, required=False)
    parser.add_argument("-t", "--report_type", help=report_type_help, default="OP")

    return parser.parse_args()


def parse_rel_weights(rel_weights_path: os.path):
    rel_weights = dict()
    rel_weights_file_path = rel_weights_path
    if rel_weights_path:
        if os.path.isdir(rel_weights_path):
            rel_weights_file_path = os.path.join(rel_weights_path, REL_WEIGHTS_FILENAME)
        if os.path.isfile(rel_weights_file_path):
            logger.info(f"Rel weights will be taken from {rel_weights_file_path}")
            with open(rel_weights_path, "r") as rel_weights_file:
                for line in rel_weights_file.readlines():
                    sep_pos = line.find(':')
                    op_name = line[:sep_pos:]
                    op_weight = float(line[sep_pos+1::].replace('\n', ''))
                    rel_weights.update({op_name: op_weight})
        else:
            logger.warning(f"Rel weights file does not exist! The expected passrates will be taken from runtime")
    else:
        logger.warning(f"Rel weights file is not specified! The expected passrates will be taken from runtime")

    return rel_weights


def merge_xmls(xml_paths: list):
    logger.info("Merging XML files is started")

    summary = Element("report")
    timestamp = None
    summary_results = SubElement(summary, "results")
    ops_list = SubElement(summary, "ops_list")
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
                op_node = SubElement(ops_list, op.tag)
                for op_attrib in op.attrib:
                    op_node.set(op_attrib, op.get(op_attrib))

        for device in xml_root.find("results"):
            device_results = summary_results.find(device.tag)
            if device_results is None:
                summary_results.append(device)
            else:
                for op_result in device:
                    current_op_res = device_results.find(op_result.tag)
                    stat_update_utils.update_rel_values(current_op_res)
                    if current_op_res is not None:
                        # workaround for unsaved reports
                        total_tests_count_xml, total_tests_count_summary = (0, 0)
                        for attr_name in device_results.find(op_result.tag).attrib:
                            if "relative_" in attr_name or attr_name == "passrate" or attr_name == "implemented":
                                continue
                            total_tests_count_xml += int(op_result.attrib.get(attr_name))
                            total_tests_count_summary += int(current_op_res.attrib.get(attr_name))
                        if total_tests_count_xml > total_tests_count_summary:
                            logger.warning(f'Test counter is different in {op_result.tag} for {device.tag}'\
                                           f'({total_tests_count_xml} vs {total_tests_count_xml})')
                            for attr_name in device_results.find(op_result.tag).attrib:
                                if attr_name == "passrate" or attr_name == "implemented" or attr_name == "relative_passrate":
                                    continue
                                xml_value = None
                                if "relative_" in attr_name:
                                    value = op_result.attrib.get(attr_name)
                                    if value is None:
                                        continue
                                    xml_value = float(op_result.attrib.get(attr_name))
                                else:
                                    xml_value = int(op_result.attrib.get(attr_name))
                                device_results.find(current_op_res.tag).set(attr_name, str(xml_value))
                    else:
                        device_results.append(op_result)
    stat_update_utils.update_passrates(summary_results)
    summary.set("timestamp", timestamp)
    logger.info("Merging XML files is competed")
    return summary


def collect_statistic(root: Element, is_conformance_mode: bool):
    logger.info("Statistic collecting is started")
    trusted_ops = dict()
    pass_rate_avg = dict()
    pass_rate_avg_rel = dict()
    general_pass_rate = dict()
    general_pass_rate_rel = dict()
    general_test_count = dict()
    general_test_count_rel = dict()
    general_passed_tests = dict()
    general_passed_tests_rel = dict()
    op_res = dict()

    results = dict()
    covered_ops = dict()
    for device in root.find("results"):
        results[device.tag] = {op.tag: op.attrib for op in device}

        pass_rate_avg[device.tag] = 0
        pass_rate_avg_rel[device.tag] = 0
        general_test_count[device.tag] = 0
        general_test_count_rel[device.tag] = 0
        general_passed_tests[device.tag] = 0
        general_passed_tests_rel[device.tag] = 0
        trusted_ops[device.tag] = 0
        covered_ops[device.tag] = 0
        for op in results[device.tag]:
            # for correct display of reports without hanged item in report.xml
            results[device.tag][op]["hanged"] = results[device.tag][op].get("hanged", 0)
            op_test_cnt = int(results[device.tag][op]["passed"]) + int(results[device.tag][op]["failed"]) + \
                          int(results[device.tag][op]["crashed"]) + int(results[device.tag][op]["skipped"]) + \
                          int(results[device.tag][op]["hanged"])
            if op_test_cnt == 0:
                continue
            covered_ops[device.tag] += 1
            pass_rate = float("%.2f"%float(results[device.tag][op]["passrate"]))
            relative_pass_rate = float("%.2f"%float(results[device.tag][op]["relative_passrate"]))
            results[device.tag][op]["passrate"] = pass_rate
            results[device.tag][op]["relative_passrate"] = relative_pass_rate

            if pass_rate == 100.:
                trusted_ops[device.tag] += 1
            device_general_test_count = op_test_cnt
            general_test_count[device.tag] += device_general_test_count
            general_test_count_rel[device.tag] += float(results[device.tag][op]["relative_all"])
            general_passed_tests[device.tag] += int(results[device.tag][op]["passed"])
            general_passed_tests_rel[device.tag] += float(results[device.tag][op]["relative_passed"])
            pass_rate_avg[device.tag] += float(results[device.tag][op]["passrate"])
            pass_rate_avg_rel[device.tag] += float(results[device.tag][op]["relative_passrate"])

            if op in op_res.keys():
                op_res[op].update({device.tag: device_general_test_count})
            else:
                op_res.update({op: {device.tag: device_general_test_count}})
        pass_rate_avg[device.tag] = 0 if covered_ops[device.tag] == 0 else pass_rate_avg[device.tag] / covered_ops[device.tag]
        pass_rate_avg[device.tag] = float("%.2f"%float(pass_rate_avg[device.tag]))
        pass_rate_avg_rel[device.tag] = 0 if covered_ops[device.tag] == 0 else pass_rate_avg_rel[device.tag] / covered_ops[device.tag]
        pass_rate_avg_rel[device.tag] = float("%.2f"%float(pass_rate_avg_rel[device.tag]))
        general_pass_rate[device.tag] = 0 if general_test_count[device.tag] == 0 else (general_passed_tests[device.tag] * 100 / general_test_count[device.tag])
        general_pass_rate[device.tag] = float("%.2f"%float(general_pass_rate[device.tag]))
        general_pass_rate_rel[device.tag] = 0 if general_test_count_rel[device.tag] == 0 else (general_passed_tests_rel[device.tag] * 100 / general_test_count_rel[device.tag])
        general_pass_rate_rel[device.tag] = float("%.2f"%float(general_pass_rate_rel[device.tag]))
        trusted_ops[device.tag] = float("%.2f"%(float("%.2f"%(float(trusted_ops[device.tag]) * 100)) / covered_ops[device.tag])) if device.tag in covered_ops and covered_ops[device.tag] != 0 else 0

    logger.info("Test number comparison between devices is started")
    for op in op_res:
        op_counter = None
        is_not_printed = True
        max_test_cnt = 0
        for dev in op_res[op]:
            if op_counter is None:
                op_counter = op_res[op][dev]
            elif op_counter != op_res[op][dev]:
                max_test_cnt = max(max_test_cnt, op_res[op][dev])
                if is_not_printed:
                    is_not_printed = False
                    logger.warning(f'{op} : {op_res[op]}')

    logger.info("Test number comparison between devices is completed")

    devices = results.keys()
    logger.info("Statistic collecting is completed")
    return devices, results, general_pass_rate, general_pass_rate_rel, pass_rate_avg, pass_rate_avg_rel, general_test_count, trusted_ops, covered_ops


def format_string(input_str: str):
    res = input_str
    res = res.replace('{', '')
    res = res.replace('}', '')
    res = res.replace("'", '')
    res = res.replace('"', '')
    res = res.replace(': ', '=')
    res = res.replace(' ', '')
    res = res.replace(',', ' ')
    return res


def serialize_to_csv(report_filename: str, output_dir: os.path, op_list: list, device_list: list, results: dict):
    csv_filename = os.path.join(output_dir, report_filename + '.csv')
    with open(csv_filename, "w", newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file, dialect='excel')
        # csv_writer.writerow(['Operation'] + device_list)
        devices_csv = ['Operation']
        device_res_csv = ['Operation']
        
        for device in device_list:          
            for status in STATUS_CSV_ORDER:
                devices_csv.append(device)
                device_res_csv.append(status)
            
        csv_writer.writerow(devices_csv)
        csv_writer.writerow(device_res_csv)

        for op in op_list:
            list_to_csv = list()
            for device in device_list:
                if op in results[device]:
                    if results[device][op] == NA or results[device][op] == NOT_RUN:
                        for status in STATUS_CSV_ORDER:
                            list_to_csv.append(results[device][op])
                        continue
                    for status in STATUS_CSV_ORDER:
                        list_to_csv.append(str(results[device][op][status]))
                else:
                    for status in STATUS_CSV_ORDER:
                        list_to_csv.append(NA)
            csv_writer.writerow([op] + list_to_csv)

    logger.info(f'Final CSV report is saved to {csv_filename}')


def create_summary(summary_root: Element, output_folder: os.path, expected_devices:list, report_tag: str, report_version: str,
                   is_conformance_mode: bool,  is_serialize_to_csv: bool, rel_weights_path: str, output_filename='report'):
    rel_weights = dict()
    if is_conformance_mode:
        stat_update_utils.update_conformance_test_counters(summary_root)
        rel_weights = parse_rel_weights(rel_weights_path)
        stat_update_utils.update_passrates(summary_root.find("results"), rel_weights)
    device_list, results, general_pass_rate, general_pass_rate_rel, pass_rate_avg, pass_rate_avg_rel, general_test_count, trusted_ops, covered_ops = \
        collect_statistic(summary_root, is_conformance_mode)

    op_list = dict()
    for op in summary_root.find("ops_list"):
        try:
            opsets = op.attrib.get("opsets").split()
            opsets = [int(opset) for opset in opsets]
        except:
            opsets = []
        op_list.update({op.tag: opsets})
    
    if len(expected_devices) > 0 and sorted(expected_devices) != device_list:
        for expected_device in expected_devices:
            if expected_device in device_list:
                continue
            tmp_res = dict()
            no_run_val = "NOT RUN"
            tmp_res = {op: no_run_val for op in op_list}
            results[expected_device] = tmp_res
            general_pass_rate[expected_device] = no_run_val
            pass_rate_avg[expected_device] = no_run_val
            general_test_count[expected_device] = no_run_val
            trusted_ops[expected_device] = no_run_val
            covered_ops[expected_device] = no_run_val
        device_list = results.keys()

    timestamp = summary_root.attrib["timestamp"]

    device_list = sorted(device_list)

    script_dir, _ = os.path.split(os.path.abspath(__file__))
    file_loader = FileSystemLoader(os.path.join(script_dir, 'template'))
    env = Environment(loader=file_loader)
    template = env.get_template('report_template.html')

    res_summary = template.render(ordered_ops=op_list, devices=device_list, results=results, timestamp=timestamp,
                                  general_pass_rate=general_pass_rate, general_pass_rate_rel=general_pass_rate_rel,
                                  pass_rate_avg=pass_rate_avg, pass_rate_avg_rel=pass_rate_avg_rel,
                                  trusted_ops=trusted_ops, covered_ops=covered_ops,
                                  general_test_count=general_test_count, report_tag=report_tag, report_version=report_version)

    report_path = os.path.join(output_folder, f'{output_filename}.html')
    with open(report_path, "w") as f:
        logger.info(f'Final report is saved to {report_path}')
        f.write(res_summary)
    if is_serialize_to_csv:
        serialize_to_csv(output_filename, output_folder, op_list, device_list, results)

def create_api_summary(xml_paths: list, output_folder: str, expected_devices:list, report_tag: str, report_version: str,
                       output_filename='report'):
        timestamp = None

        api_info = {}
        sw_plugins = ['MULTI', 'BATCH', 'AUTO', 'HETERO']
        api_devices = set(expected_devices) if expected_devices else set()

        logger.info("Statistic collecting is started")
        for xml_path in xml_paths:
            if not Path(xml_path).exists():
                logger.error(f'File is not exists: {xml_path}')
                continue
            try:
                xml_root = ET.parse(xml_path).getroot()
                if timestamp is None or timestamp < xml_root.attrib["timestamp"]:
                    timestamp = xml_root.attrib["timestamp"]

                for device in xml_root.findall("results/*"):
                    if expected_devices and device.tag not in expected_devices:
                        continue

                    api_devices.add(device.tag)
                    for test_type in xml_root.findall(f"results/{device.tag}/*"):
                        api_info.setdefault(test_type.tag, {})
                        for plugin_info in xml_root.findall(f"results/{device.tag}/{test_type.tag}/*"):
                            if str(plugin_info.tag).upper() != str(device.tag).upper() and\
                               (plugin_info.tag in sw_plugins and str(plugin_info.tag).upper() != 'TEMPLATE'):
                                continue

                            api_info[test_type.tag].setdefault(device.tag, {'test_amout': 0})
                            for key in ['passed', 'failed', 'crashed', 'skipped', 'hanged', 'relative_all', 'relative_passed']:
                                val = int(float(plugin_info.get(key, 0)))
                                api_info[test_type.tag][device.tag]['test_amout'] += val if 'relative' not in key else 0
                                val += api_info[test_type.tag][device.tag].setdefault(key, 0)
                                api_info[test_type.tag][device.tag][key] = val

                            if api_info[test_type.tag][device.tag]['relative_all'] > 0:
                                api_info[test_type.tag][device.tag]['relative_passrate'] = round(api_info[test_type.tag][device.tag]['relative_passed'] * 100 /\
                                                                                                 api_info[test_type.tag][device.tag]['relative_all'], 2)
                            if api_info[test_type.tag][device.tag]['test_amout'] > 0:
                                api_info[test_type.tag][device.tag]['passrate'] = round(api_info[test_type.tag][device.tag]['passed'] * 100 /\
                                                                                        api_info[test_type.tag][device.tag]['test_amout'], 2)

            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')
        logger.info("Statistic collecting is completed")

        logger.info("File with report creating is started")
        script_dir = Path(__file__).parent.absolute()
        file_loader = FileSystemLoader(script_dir.joinpath('template').as_posix())
        env = Environment(loader=file_loader)
        template = env.get_template('report_api_template.html')

        res_summary = template.render(devices=api_devices,
                                      api_info=api_info,
                                      timestamp=timestamp,
                                      report_tag=report_tag,
                                      report_version=report_version)

        report_path = Path()
        if output_folder and Path(output_folder).is_dir():
            report_path = Path(output_folder)

        report_path = report_path.joinpath(f'{output_filename}.html')

        with open(report_path.as_posix(), "w") as f:
            logger.info(f'Final report is saved to {report_path}')
            f.write(res_summary)

if __name__ == "__main__":
    args = parse_arguments()
    if args.report_type == 'OP':
        summary_root = merge_xmls(args.xml)
        create_summary(summary_root, args.out,
                    [] if args.expected_devices is None else args.expected_devices,
                    args.report_tag,
                    args.report_version,
                    args.conformance_mode,
                    args.csv,
                    args.rel_weights,
                    args.output_filename)
    else:
        create_api_summary(args.xml, args.out, args.expected_devices,
                           args.report_tag, args.report_version, args.output_filename)

