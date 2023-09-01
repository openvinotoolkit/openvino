# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
import os
import importlib


skip_snippets = ["main.py", "__init__.py", "ie_common.py", "ov_common.py"]

def import_python_modules(directory, subdirectory=""):
    for item in os.listdir(directory):
        if item.endswith('.py') and item not in skip_snippets:
            imported_item = item[:-3]
            if subdirectory != "":
                imported_item=subdirectory + "." + imported_item
            mod = importlib.import_module(imported_item)
            try:
                mod.main()
            except AttributeError as e:
                pass
        
        if os.path.isdir(os.path.join(directory, item)):
            dir_path = os.path.join(directory, item)
            import_python_modules(dir_path, item)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import_python_modules(dir_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
