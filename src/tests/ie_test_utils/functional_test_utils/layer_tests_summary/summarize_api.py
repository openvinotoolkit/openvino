import os
import re
import sys
import argparse

from utils.conformance_utils import get_logger

from pathlib import Path
import xml.etree.ElementTree as ET
from jinja2 import Environment, FileSystemLoader


logger = get_logger('HighlightTable')

OPSET_REPORT_NAME_RE = r'.*report_opset_\w*.xml'
API_REPORT_NAME_RE = r'.*report_api_\w*.xml'


def parse_arguments():
    parser = argparse.ArgumentParser()

    xml_help = """
        Paths to folder with xml summary files.
        Script analyze xml files with name report_opset_[plugin].xml and 
        report_api_[plugin].xml.
        Folder may have any structure, but if you setup expected_test_mode, 
        it is need to have folder name as test_mode in folder strucutre.
    """
    report_tag_help = "Report tag"
    report_version_help = "Report version"
    output_filename_help = "Output report file"
    output_folder_help = "Output report folder"
    expected_devices_help = "List of expected devices"

    parser.add_argument("--xml", help=xml_help, nargs="*", required=True)
    parser.add_argument("--output_folder", help=output_folder_help)
    parser.add_argument("--output_filename", help=output_filename_help, default="highlight_table.html")
    parser.add_argument("--expected_devices", help=expected_devices_help, nargs="*", required=False)

    parser.add_argument("--report_tag", help=report_tag_help, default="")
    parser.add_argument("--report_version", help=report_version_help, default="")


    return parser.parse_args()

def collect_xml_pathes(xmls):
    opset_xmls = []
    api_xmls = []

    for xml in list(Path(xmls).glob('**/*.xml')):
        if re.match(OPSET_REPORT_NAME_RE, xml.as_posix()):
            opset_xmls.append(xml)
        elif re.match(API_REPORT_NAME_RE, xml.as_posix()):
            api_xmls.append(xml)

    return opset_xmls, api_xmls


class APISammaryCreator():
    def __init__(self, xmls_paths, expected_devices=None) -> None:
        self.xmls_paths = xmls_paths

        self.devices = expected_devices if expected_devices else []
        self.expected_devices = expected_devices
        self.sw_plugins = set()

        self.api_info = {}
        self.timestamp = None

    def get_ops_pass_statictic(self, xml_root, device):
        passed_tests = 0
        test_count_test = 0
        total_passed_tests = 0
        total_amount_ops = 0
        relative_passrate = 0
        total_passrate = 0

        rel_passed = 0
        rel_all = 0
        for op in xml_root.findall(f'results/{device}/*'):
            relative_passrate += float(op.get("relative_passrate", 0))
            total_passrate += float(op.get("passrate", 0))

            rel_passed += float(op.get("relative_passed", 0))
            rel_all += float(op.get("relative_all", 0))

            count = int(op.get("passed")) + int(op.get("failed")) + \
                    int(op.get("crashed")) + int(op.get("skipped")) + \
                    int(op.get("hanged"))
            test_count_test += count
            if count > 0:
                total_amount_ops += 1

            passed_tests += int(op.get("passed"))

        return passed_tests, total_passed_tests, test_count_test, total_amount_ops, relative_passrate, total_passrate, rel_passed, rel_all

    def update_real_devices(self, devices):
        # for case when expected devices is not set and script get devices from xmls
        not_considered_devices = devices.difference(set(self.devices))
        self.devices.extend(list(not_considered_devices))

    def build_sw_plugin_name(self, sw_plugin, device):
        return 'HW_PLUGIN' if str(sw_plugin).upper() == str(device).upper() else sw_plugin

    def collect_api_info(self):
        logger.info("API info collecting is started")

        api_devices = set()
        for xml_path in self.xmls_paths:
            if not Path(xml_path).exists():
                logger.error(f'File is not exists: {xml_path}')
                continue
            try:
                xml_root = ET.parse(xml_path).getroot()
                if self.timestamp is None or self.timestamp < xml_root.attrib["timestamp"]:
                    self.timestamp = xml_root.attrib["timestamp"]

                for device in xml_root.findall("results/*"):
                    if self.expected_devices and device.tag not in self.expected_devices:
                        continue

                    api_devices.add(device.tag)
                    for test_type in xml_root.findall(f"results/{device.tag}/*"):
                        self.api_info.setdefault(test_type.tag, {})
                        for sw_plugin in xml_root.findall(f"results/{device.tag}/{test_type.tag}/*"):
                            sw_plugin_name = self.build_sw_plugin_name(sw_plugin.tag, device.tag)
                            self.sw_plugins.add(sw_plugin_name)
                            self.api_info[test_type.tag].setdefault(sw_plugin_name, {device.tag: {}})
                            self.api_info[test_type.tag][sw_plugin_name][device.tag] = {'passrate': float(sw_plugin.get('passrate', 0)),
                                                                                        "relative_passrate": float(sw_plugin.get('relative_passrate', 0)),
                                                                                        "relative_all": float(sw_plugin.get('relative_all', 0)),
                                                                                        "relative_passed": float(sw_plugin.get('relative_passed', 0)),
                                                                                        'passed': int(sw_plugin.get('passed', 0)), 
                                                                                        'failed': int(sw_plugin.get('failed', 0)),
                                                                                        'crashed': int(sw_plugin.get('crashed', 0)),
                                                                                        'skipped': int(sw_plugin.get('skipped', 0)),
                                                                                        'hanged': int(sw_plugin.get('hanged', 0)),
                                                                                        'test_amout': int(sw_plugin.get('passed', 0)) +\
                                                                                                      int(sw_plugin.get('failed', 0)) +\
                                                                                                      int(sw_plugin.get('passed', 0)) +\
                                                                                                      int(sw_plugin.get('crashed', 0)) +\
                                                                                                      int(sw_plugin.get('hanged', 0))}

            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')

        self.update_real_devices(api_devices)

    def create_html(self, output_folder=None, output_filename=None, report_tag="", report_version=""):
        sw_plugins = list(self.sw_plugins)
        sw_plugins.sort()
        if 'HW_PLUGIN' in sw_plugins:
            sw_plugins.remove('HW_PLUGIN')
            sw_plugins.insert(0, 'HW_PLUGIN')

        script_dir = Path(__file__).parent.absolute()
        file_loader = FileSystemLoader(script_dir.joinpath('template').as_posix())
        env = Environment(loader=file_loader)
        template = env.get_template('report_api_template.html')

        res_summary = template.render(devices=self.devices,
                                      api_info=self.api_info,
                                      sw_plugins=sw_plugins,
                                      timestamp=self.timestamp,
                                      report_tag=report_tag,
                                      report_version=report_version)

        report_path = Path()
        if output_folder and Path(output_folder).is_dir():
            report_path = Path(output_folder)

        if output_filename:
            report_path = report_path.joinpath(output_filename)

        with open(report_path.as_posix(), "w") as f:
            f.write(res_summary)

if __name__ == "__main__":

    args = parse_arguments()

    table = APISammaryCreator(args.xml,
                                  args.expected_devices)

    table.collect_api_info()

    table.create_html(args.output_folder, args.output_filename, args.report_tag, args.report_version)
