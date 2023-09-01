# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
import os
import importlib

import openvino as ov

#skip_snippets = ["main.py", "__init__.py"]
skip_snippets = ["main.py", "__init__.py", "ie_common.py", "ov_python_exclusives.py", "ov_preprocessing.py"]

def import_python_modules(directory, subdirectory=""):
    for item in os.listdir(directory):
        if item.endswith('.py') and item not in skip_snippets:
            imported_item = item[:-3]
            if subdirectory != "":
                imported_item=subdirectory + "." + imported_item
            try:
                mod = importlib.import_module(imported_item)
            except RuntimeError as e:

            try:
                mod.main()
            except:
                print("fail")
                pass
        
        if os.path.isdir(os.path.join(directory, item)):
            dir_path = os.path.join(directory, item)
            import_python_modules(dir_path, item)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import_python_modules(dir_path)

    # import ov_caching
    # try:
    #     core = ov.Core()
    #     available_devices = core.available_devices
    #     if "GPU" in available_devices:
    #         print("run gpu snippets")

    # except Exception as e:
    #     sys.exit(1)



if __name__ == "__main__":
    sys.exit(main() or 0)
