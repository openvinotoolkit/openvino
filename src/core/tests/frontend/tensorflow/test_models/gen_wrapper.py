# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import sys

print(sys.argv)
if len(sys.argv) < 4:
    print("Script, output folder and mark file must be specified as arguments")
    exit(1)

gen_script = sys.argv[1]
out_folder = sys.argv[2]
mark_file = sys.argv[3]

print("Processing: {} ".format(gen_script))
subprocess.run([sys.executable, gen_script, out_folder], env=os.environ)

# Create mark file indicating that script was executed
with open(mark_file, "w") as fp:
    pass
