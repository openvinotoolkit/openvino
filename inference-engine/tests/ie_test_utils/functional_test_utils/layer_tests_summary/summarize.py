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


def merge_xmls(xmls):
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
    xmls.append(ET.parse(xml).getroot())

root = merge_xmls(xmls)
timestamp = root.attrib["timestamp"]
ops = []
for op in root.find("ops_list"):
    ops.append(op.tag)
ordered_ops = sorted(ops)
results = {}
for device in root.find("results"):
    results[device.tag] = {op.tag: op.attrib for op in device}
    for op in results[device.tag]:
        results[device.tag][op]["passrate"] = round(float(results[device.tag][op]["passrate"]), 1)

devices = results.keys()

file_loader = FileSystemLoader('template')
env = Environment(loader=file_loader)
template = env.get_template('report_template.html')

res = template.render(ordered_ops=ordered_ops, devices=devices, results=results, timestamp=timestamp)

with open(os.path.join(args.out, "report.html"), "w") as f:
    f.write(res)
