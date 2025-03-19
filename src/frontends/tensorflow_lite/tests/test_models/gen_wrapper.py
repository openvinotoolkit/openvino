# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

if len(sys.argv) < 4:
    print("Script[model in pbtxt format], output folder and mark file must be specified as arguments", str(sys.argv))
    exit(1)

gen_script = sys.argv[1]
out_folder = sys.argv[2]
mark_file = sys.argv[3]

assert gen_script.endswith('.py'), "Unexpected script: " + gen_script
try:
    os.makedirs(out_folder, exist_ok=True)
    subprocess.run([sys.executable, gen_script, out_folder], env=os.environ, check=True)
except subprocess.CalledProcessError as cpe:
    if cpe.returncode == 2:
        exit(2)

# Create mark file indicating that script was executed
with open(mark_file, "w") as fp:
    pass
