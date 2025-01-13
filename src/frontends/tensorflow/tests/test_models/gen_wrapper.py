# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys

# do not print messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if len(sys.argv) < 4:
    print("Script[model in pbtxt format], output folder and mark file must be specified as arguments", str(sys.argv))
    exit(1)

gen_script = sys.argv[1]
out_folder = sys.argv[2]
mark_file = sys.argv[3]

if gen_script.endswith('.py'):
    subprocess.run([sys.executable, gen_script, out_folder], env=os.environ)
elif gen_script.endswith('.pbtxt'):
    model_pbtxt = gen_script
    model_name = os.path.basename(model_pbtxt).split('.')[0]
    dest_path = os.path.join(out_folder, model_name, model_name + '.pbtxt')
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        shutil.copy(model_pbtxt, dest_path)
    except shutil.SameFileError:
        pass

# Create mark file indicating that script was executed
with open(mark_file, "w") as fp:
    pass
