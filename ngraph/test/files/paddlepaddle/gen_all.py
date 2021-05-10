import os
import subprocess

import sys
import glob

if len(sys.argv) < 2:
    print("Output models folder must be specified as first argument")
    exit(1)

gen_dir_name = os.path.dirname(os.path.realpath(__file__))
print(gen_dir_name)
for gen_script in glob.glob(os.path.join(gen_dir_name, "gen_scripts", "generate_*.py")):
    print("Processing: {} ".format(gen_script))
    subprocess.run([sys.executable, gen_script, *sys.argv[1:]], env=os.environ)
