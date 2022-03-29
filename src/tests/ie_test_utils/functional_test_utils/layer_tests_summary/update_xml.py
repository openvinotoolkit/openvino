import re
import os
import logging

from xml.dom import minidom
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


STATUSES = {'SKIPPED': 'skipped', 'OK': 'passed', 'FAILED': 'failed', 'HANGED': 'hanged', 'CRASHED': 'crashed'}


class XMLUpdater():
    def __init__(self, path_to_folder, xml_name, device):
        self.path_to_folder = path_to_folder
        self.xml_name = xml_name
        self.device = device
        self.xml_path = os.path.join(path_to_folder, xml_name)
        self.logger = logging.getLogger("xmlUpdaterLoger")

    def collect_ops_info(self, output_str: str):
        ops_info = {}

        test_ops = []
        test_name = ''
        for line in output_str.split("\n"):
            # remove Unicode whitespace characters from the end of str
            line = re.sub(r"\s*$", "", line)

            match = re.match(r"\[ RUN      \] (.*)\s*", line)
            if match:
                test_name = match[1].strip()

            # example: OPS LIST: Convolution-1 Add-1
            match = re.match(r"OPS LIST: (.*)", line)
            if match:
                test_ops = match[1].split(" ")

            # example: [       OK ] conformance/ReadIRTest.ReadIR/PRC=f32_IR_name=Tanh_614446.xml_TargetDevice=CPU_Config=() (83 ms)
            match = re.match(r"\[\s*(SKIPPED|OK|FAILED)\s*\] .* \(\d* ms\)\s*", line)
            if match:
                status = STATUSES[match[1]]
                for op in test_ops:
                    if op not in ops_info:
                        ops_info[op] = {}
                    ops_info[op][status] = 0 if 'OpImplCheckTest' in test_name else ops_info[op].get(status, 0) + 1

                test_ops = []

        if (test_ops):
            for op in test_ops:
                if op not in ops_info:
                    ops_info[op] = {}
                ops_info[op]['crashed'] = ops_info[op].get('crashed', 0) + 1

        return ops_info

    def update_xml(self, ops_info: dict):
        try:
            xml_root = ET.parse(self.xml_path).getroot()
        except ET.ParseError:
            self.logger.error(f' {self.xml_path} is corrupted and skipped')
            return None

        device_tag = xml_root.find("results").find(self.device)
        if device_tag is None:
            device_tag = ET.SubElement(xml_root, self.device)
            device_tag.tail = '\n\t'

        for op_name, op_info in ops_info.items():
            op_tag = device_tag.find(op_name)
            if op_tag is None:
                op_tag = ET.SubElement(device_tag, op_name)
                op_tag.tail = '\n\t\t'
                op_tag.set("implemented", "true")
                op_tag.set("passed", "0")
                op_tag.set("failed", "0")
                op_tag.set("skipped", "0")
                op_tag.set("crashed", "0")
                op_tag.set("hanged", "0")
                op_tag.set("passrate", "0")

                test_ops = xml_root.find("ops_list")
                if test_ops.find(op_name) is None:
                    op_name_tag = ET.SubElement(test_ops, op_name)
                    op_name_tag.tail = '\n\t'

            for name, val in op_info.items():
                old_val = op_tag.attrib.get(name)
                new_val = int(val) + int(old_val)
                op_tag.set(name, str(new_val))

        return xml_root

    def create_xml(self, ops_info: dict):
        summary = ET.Element("report")
        summary.set('timestamp',  datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        results = ET.SubElement(summary, "results")
        test_ops = ET.SubElement(summary, "ops_list")

        device_tag = ET.SubElement(results, self.device)

        for op_name, op_info in ops_info.items():
            op_name_tag = ET.SubElement(test_ops, op_name)

            op_tag = ET.SubElement(device_tag, op_name)
            op_tag.set("implemented", "true")
            op_tag.set("passed", "0")
            op_tag.set("failed", "0")
            op_tag.set("skipped", "0")
            op_tag.set("crashed", "0")
            op_tag.set("hanged", "0")
            op_tag.set("passrate", "0")
            
            for name, val in op_info.items():
                op_tag.set(name, str(val))
        
        return summary

    def update_xml_file(self, xml_root, pretty_xml=False):
        if not os.path.isdir(self.path_to_folder):
            self.logger.error(f" {self.path_to_folder} is not a directory!")
            return 1

        try:
            if not os.path.exists(self.path_to_folder):
                os.makedirs(self.path_to_folder)

            if pretty_xml:
                xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent='\t', newl='\n')
                with open(self.xml_path, "w") as xml_file:
                    xml_file.write(xmlstr)
            else:
                with open(self.xml_path, "w") as xml_file:
                    ET.ElementTree(xml_root).write(self.xml_path)

        except EnvironmentError as e:
            self.logger.error(f" Error: {e.__class__} occurred in writing xml data to {self.xml_path}!")
            return 1

        return 0

    def process_result(self, output):
        ops_info = self.collect_ops_info(output)
        if not ops_info:
            self.logger.warning(f" Could not get info from output !")
            return 1

        pretty_xml = False
        if os.path.exists(self.xml_path):
            xml_root = self.update_xml(ops_info)
            if xml_root is None:
                self.logger.warning(f" Could not update xml data !")
                return 1
        else:
            xml_root = self.create_xml(ops_info)
            pretty_xml = True
            if xml_root is None:
                self.logger.warning(f" Could not create xml date !")
                return 1

        res = self.update_xml_file(xml_root, pretty_xml)

        return res
