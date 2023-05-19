import os
import re
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
    output_filename_help = "Output report file"
    output_folder_help = "Output report folder"
    expected_devices_help = "List of expected devices"
    expected_test_mode_help = """
        Test mode like static, dymanic or apiConformance,
        it will be defined by path
        If script will found xml, but path will not include test_mode,
        script will save result in Other.
    """

    parser.add_argument('--current_xmls', help=xml_help, required=True)
    parser.add_argument('--prev_xmls', help=xml_help, required=True)

    parser.add_argument("--output_folder", help=output_folder_help)
    parser.add_argument("--output_filename", help=output_filename_help, default="highlight_table.html")
    parser.add_argument("--expected_devices", help=expected_devices_help, nargs="*", required=False)
    parser.add_argument("--expected_test_mode", help=expected_test_mode_help, nargs="*", required=False)

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


class HighlightTableCreator():
    def __init__(self,
                 current_xmls_opset,
                 prev_xmls_opset,
                 current_xmls_api,
                 prev_xmls_api,
                 expected_devices=None) -> None:

        self.current_xmls_opset = current_xmls_opset
        self.prev_xmls_opset = prev_xmls_opset
        self.current_xmls_api = current_xmls_api
        self.prev_xmls_api = prev_xmls_api

        self.devices = expected_devices if expected_devices else []
        self.expected_devices = expected_devices
        self.sw_plugins = set()

        self.test_modes = ['Other']
        self.test_modes_re = ''

        self.ops_info = {} 
        self.general_pass_rate = {}

        self.api_info = {}


    def setup_test_modes(self, expected_test_modes):
        self.test_modes = expected_test_modes + self.test_modes
        self.test_modes_re = '|'.join(expected_test_modes)
        logger.info(f'Test modes {self.test_modes}')

    def get_test_mode_by_path(self, xml_path):
        # Other
        test_mode = self.test_modes[-1]
        test_mode_match = None
        if self.test_modes_re:
            test_mode_match = re.match(f'.*({self.test_modes_re}).*', xml_path.as_posix())
        if test_mode_match:
            test_mode = test_mode_match.group(1)

        return test_mode

    def get_ops_pass_statictic(self, xml_root, device):
        passed_tests = 0
        test_count_test = 0
        total_passed_tests = 0
        total_amount_ops = 0
        for op in xml_root.findall(f'results/{device}/*'):
            if op.get('passrate', '0') == '100':
                total_passed_tests += 1
            count = int(op.get("passed")) + int(op.get("failed")) + \
                    int(op.get("crashed")) + int(op.get("skipped")) + \
                    int(op.get("hanged"))
            test_count_test += count
            if count > 0:
                total_amount_ops += 1

            passed_tests += int(op.get("passed"))

        return passed_tests, total_passed_tests, test_count_test, total_amount_ops

    def get_general_passrate(self, test_count, passed_tests):
        general_pass_rate = 0 if test_count == 0 else (passed_tests * 100 / test_count)
        general_pass_rate = round(float(general_pass_rate), 1)
        return general_pass_rate

    def update_real_devices(self, devices):
        # for case when expected devices is not set and script get devices from xmls
        not_considered_devices = devices.difference(set(self.devices))
        self.devices.extend(list(not_considered_devices))

    def collect_opset_info(self):
        logger.info("Opset info collecting is started")

        ops_devices = set()
        for test_mode in self.test_modes:
            self.ops_info[test_mode] = {}
            self.general_pass_rate[test_mode] = {}

        for xml_path in self.current_xmls_opset:
            test_mode = self.get_test_mode_by_path(xml_path)
            try:
                xml_root = ET.parse(xml_path).getroot()
                for device in xml_root.findall("results/*"):
                    if self.expected_devices and device.tag not in self.expected_devices:
                        continue

                    ops_devices.add(device.tag)
                    passed_tests, total_passed_tests, test_count, total_amount_ops = self.get_ops_pass_statictic(xml_root, device.tag)

                    self.ops_info[test_mode][device.tag] = {'totalAmount': total_amount_ops,
                                                            'diffTotalAmount': 0, 
                                                            'totalPass': total_passed_tests,
                                                            'diffTotalPass': 0}

                    self.general_pass_rate[test_mode][device.tag] = {'current': self.get_general_passrate(test_count, passed_tests), 'prev': 0}

            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')

        for xml_path in self.prev_xmls_opset:
            test_mode = self.get_test_mode_by_path(xml_path)
            try:
                xml_root = ET.parse(xml_path).getroot()
                for device in xml_root.findall("results/*"):
                    if device.tag not in ops_devices:
                        continue

                    passed_tests, total_passed_tests, test_count, total_amount_ops = self.get_ops_pass_statictic(xml_root, device.tag)

                    self.ops_info[test_mode][device.tag]['diffTotalAmount'] = self.ops_info[test_mode][device.tag]['totalAmount'] - total_amount_ops
                    self.ops_info[test_mode][device.tag]['diffTotalPass'] = self.ops_info[test_mode][device.tag]['totalPass'] - total_passed_tests
                    self.general_pass_rate[test_mode][device.tag]['prev'] = round(self.general_pass_rate[test_mode][device.tag]['current'] -\
                                                                                   self.get_general_passrate(test_count, passed_tests), 1)

            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')

        self.update_real_devices(ops_devices)

    def build_sw_plugin_name(self, sw_plugin, device):
        return 'HW PLUGIN' if str(sw_plugin).upper() == str(device).upper() else sw_plugin

    def collect_api_info(self):
        logger.info("API info collecting is started")

        api_devices = set()
        for xml_path in self.current_xmls_api:
            try:
                xml_root = ET.parse(xml_path).getroot()
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
                            self.api_info[test_type.tag][sw_plugin_name][device.tag] = {'passrate': float(sw_plugin.get('passrate')), 'diff': 0}
            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')

        for xml_path in self.prev_xmls_api:
            try:
                xml_root = ET.parse(xml_path).getroot()
                for device in xml_root.findall("results/*"):
                    if device.tag not in api_devices:
                        continue

                    for test_type in xml_root.findall(f"results/{device.tag}/*"):
                        self.api_info.setdefault(test_type.tag, {})
                        for sw_plugin in xml_root.findall(f"results/{device.tag}/{test_type.tag}/*"):
                            sw_plugin_name = self.build_sw_plugin_name(sw_plugin.tag, device.tag)
                            self.api_info[test_type.tag].setdefault(sw_plugin_name, {device.tag: {'passrate': 0, 'diff': 0}})
                            self.api_info[test_type.tag][sw_plugin_name][device.tag]['diff'] = round(self.api_info[test_type.tag][sw_plugin_name][device.tag].get('passrate', 0) -\
                                                                                                     float(sw_plugin.get('passrate')), 1)
            except ET.ParseError:
                logger.error(f'Error parsing {xml_path}')

        self.update_real_devices(api_devices)

    def create_html(self, output_folder=None, output_filename=None):
        if 'Other' in self.ops_info and self.ops_info['Other'] == {}:
            self.ops_info.pop('Other')
            self.general_pass_rate.pop('Other')
            self.test_modes.remove('Other')

        sw_plugins = list(self.sw_plugins)
        sw_plugins.sort()
        if 'HW PLUGIN' in sw_plugins:
            sw_plugins.remove('HW PLUGIN')
            sw_plugins.insert(0, 'HW PLUGIN')

        script_dir = Path(__file__).parent.absolute()
        file_loader = FileSystemLoader(script_dir.joinpath('template').as_posix())
        env = Environment(loader=file_loader)
        template = env.get_template('highlight_tables_template.html')

        res_summary = template.render(devices=self.devices,
                                      ops_info=self.ops_info,
                                      general_pass_rate=self.general_pass_rate,
                                      expected_test_mode=self.test_modes,
                                      api_info=self.api_info,
                                      sw_plugins=sw_plugins)

        report_path = Path()
        if output_folder and Path(output_folder).is_dir():
            report_path = Path(output_folder)

        if output_filename:
            report_path = report_path.joinpath(output_filename)

        with open(report_path.as_posix(), "w") as f:
            f.write(res_summary)

if __name__ == "__main__":

    args = parse_arguments()

    current_xmls_opset, current_xmls_api = collect_xml_pathes(args.current_xmls)
    prev_xmls_opset, prev_xmls_api = collect_xml_pathes(args.prev_xmls)

    if len(current_xmls_opset) == 0 and len(current_xmls_api) == 0:
        if len(current_xmls_opset) == 0:
            logger.error(f'It was not found xmls with name report_opset_[plugin].xml by path {args.current_xmls}')
        if len(current_xmls_opset) == 0:
            logger.error(f'It was not found xmls with name report_api_[].xml by path {args.prev_xmls}')
        exit(1)

    table = HighlightTableCreator(current_xmls_opset,
                                  prev_xmls_opset,
                                  current_xmls_api,
                                  prev_xmls_api,
                                  args.expected_devices)
    if args.expected_test_mode:
        table.setup_test_modes(args.expected_test_mode)


    table.collect_opset_info()
    table.collect_api_info()
    
    table.create_html(args.output_folder, args.output_filename)


