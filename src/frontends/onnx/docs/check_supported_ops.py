# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
from pathlib import Path

def run_in_ci():
    if "CI" in os.environ and os.environ["CI"].lower() == "true":
        return True

    if "TF_BUILD" in os.environ and len(os.environ["TF_BUILD"]):
        return True

    if "JENKINS_URL" in os.environ and len(os.environ["JENKINS_URL"]):
        return True

    return False


if not run_in_ci() and not '--manual' in sys.argv:
    # execute check only in CI when the code is productized
    exit(0)

if len(sys.argv) < 3:
    error_message = "Run in the following format: check_supported_ops.py path/to/ops supported_ops.md [--manual]\n"
    error_message += "  --manual - script originated to run in CI, use this flag to run it manually"
    raise Exception(error_message)

lookup_path = Path(sys.argv[1])
supported_ops_doc = sys.argv[2]

if not lookup_path.exists() or not lookup_path.is_dir():
    raise Exception(f"Argument \'{lookup_path}\' isn\'t a valid path to sources")

files = []
# Looking for source files
for path in lookup_path.rglob('*.cpp'):
    files.append(path)
for path in lookup_path.rglob('*.hpp'):
    files.append(path)

#                           MACRO        Op Name         Opset        Impl         Domain
op_regex = re.compile(r'ONNX_OP(_M)?\("([a-z0-9_]+)",\s+([^\)\}]+)[\)\}]?,\s+([a-z0-9_:]+)(,\s+[^\)]+)?\);', re.IGNORECASE)

ops = {}

known_domains = {
    "":"",
    "OPENVINO_ONNX_DOMAIN":"org.openvinotoolkit",
    "MICROSOFT_DOMAIN":"com.microsoft",
    "PYTORCH_ATEN_DOMAIN":"org.pytorch.aten",
    "MMDEPLOY_DOMAIN":"mmdeploy"
}

hdr = ""
with open(supported_ops_doc, 'rt') as src:
    table_line = 0
    for line in src:
        if table_line < 2:
            hdr += line
        if line.count('|') == 6:
            table_line += 1
        if table_line > 2:
            row = [cell.strip() for cell in line.split('|')] # Split line by "|" delimeter and remove spaces
            domain = row[1]
            if not domain in ops:
                ops[domain] = {}
            opname = row[2]
            defined = []
            for item in row[4].split(', '):
                val = 1
                try:
                    val = int(item)
                except:
                    continue
                defined.append(val)
            if not opname in ops[domain]:
                ops[domain][opname] = {'supported':[], 'defined': defined, 'limitations':row[5]}

documentation_errors = []

for file_path in files:
    with open(file_path.as_posix(), "r") as src:
        reg_macro = None
        for line in src:
            # Multiline registration
            if 'ONNX_OP' in line:
                reg_macro = ""
            if not reg_macro is None:
                reg_macro += line
            else:
                continue
            if not ');' in line:
                continue
            # Registration macro has been found, trying parse it
            m = op_regex.search(reg_macro)
            if m is None:
                documentation_errors.append(f"Registration in file {file_path.as_posix()} is corrupted {reg_macro}, please check correctness")
                if ');' in line: reg_macro = None
                continue
            domain = m.group(5)[2:].strip() if not m.group(5) is None else ""
            if not domain in known_domains:
                documentation_errors.append(f"Unknown domain found in file {file_path.as_posix()} with identifier {domain}, please modify check_supported_ops.py if needed")
                if ');' in line: reg_macro = None
                continue
            domain = known_domains[domain]
            opname = m.group(2)
            opset = m.group(3)
            if not domain in ops:
                documentation_errors.append(f"Domain {domain} is missing in a list of documented operations supported_ops.md, update it by adding operation description")
                if ');' in line: reg_macro = None
                continue
            if not opname in ops[domain]:
                documentation_errors.append(f"Operation {domain if domain=='' else domain + '.'}{opname} is missing in a list of documented operations supported_ops.md, update it by adding operation description")
                if ');' in line: reg_macro = None
                continue
            if opset.startswith('OPSET_SINCE'):
                ops[domain][opname]['supported'].append(int(opset[12:]))
            elif opset.startswith('OPSET_IN'):
                ops[domain][opname]['supported'].append(int(opset[9:]))
            elif opset.startswith('OPSET_RANGE'):
                ops[domain][opname]['supported'].append(int(opset[12:].split(',')[0]))
            elif opset.startswith('{'):
                ops[domain][opname]['supported'].append(int(opset[1:].split(',')[0]))
            else:
                documentation_errors.append(f"Domain {domain} is missing in a list of documented operations supported_ops.md, update it by adding operation description")
                if ');' in line: reg_macro = None
                continue
            if ');' in line:
                reg_macro = None

if len(documentation_errors) > 0:
    for errstr in documentation_errors:
        print('[ONNX Frontend] ' + errstr)
    raise Exception('[ONNX Frontend] failed: due to documentation errors')

with open(supported_ops_doc, 'wt') as dst:
    dst.write(hdr)
    for domain, ops in ops.items():
        for op_name in sorted(list(ops.keys())):
            data = ops[op_name]
            min_opset = data['defined'][-1] if len(data['defined']) > 0 else 1
            if min_opset in data['supported']:
                min_opset = 1
            dst.write(f"|{domain:<24}|{op_name:<56}|{', '.join([str(max(i, min_opset)) for i in sorted(data['supported'], reverse=True)]):<24}|{', '.join([str(i) for i in data['defined']]):<32}|{data['limitations']:<32}|\n")

print("Data collected and stored")
