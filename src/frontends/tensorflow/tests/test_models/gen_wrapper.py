# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

print(sys.argv)
if len(sys.argv) < 4:
    print("Script[model in pbtxt format], output folder and mark file must be specified as arguments")
    exit(1)

gen_script = sys.argv[1]
out_folder = sys.argv[2]
mark_file = sys.argv[3]

print("Processing: {} ".format(gen_script))

if gen_script.endswith('.py'):
    subprocess.run([sys.executable, gen_script, out_folder], env=os.environ)
elif gen_script.endswith('.pbtxt'):
    import tensorflow.compat.v1 as tf
    from google.protobuf import text_format

    model_pbtxt = gen_script
    with open(model_pbtxt, "r") as f:
        model_name = os.path.basename(model_pbtxt).split('.')[0]
        graph_def = tf.GraphDef()
        text_format.Merge(f.read(), graph_def)
        tf.import_graph_def(graph_def, name='')
        tf.io.write_graph(graph_def, os.path.join(sys.argv[2], model_name), model_name + '.pb', False)

# Create mark file indicating that script was executed
with open(mark_file, "w") as fp:
    pass
