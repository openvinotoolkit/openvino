import xml.etree.ElementTree as ET
from jinja2 import Environment, FileSystemLoader
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--xml", help="Path to xml summary from layer tests", required=True)
parser.add_argument("--out", help="Path to save html report", default="")
args = parser.parse_args()


file_loader = FileSystemLoader('template')
env = Environment(loader=file_loader)
template = env.get_template('report_template.html')

tree = ET.parse(args.xml)
root = tree.getroot()
timestamp = root.attrib["timestamp"]
opset = root.find("ops_list").attrib["opset_version"]
ops = {}
for op in root.find("ops_list").findall("operation"):
    ops[op.attrib['name']] = op.attrib["version"]
ordered_ops = sorted(ops.keys())
results = {}
for device in root.find("results"):
    results[device.tag] = {op.attrib['name']: op.attrib for op in device.findall("operation")}
    for op in results[device.tag]:
        results[device.tag][op]["passrate"] = round(float(results[device.tag][op]["passrate"]), 1)
        if results[device.tag][op]["version"] != ops[op]:
            results[device.tag][op]["wrong_version"] = True

devices = results.keys()

res = template.render(ops=ops, ordered_ops=ordered_ops, devices=devices, results=results, timestamp=timestamp, opset=opset)
with open(os.path.join(args.out, "report.html"), "w") as f:
    f.write(res)
pass