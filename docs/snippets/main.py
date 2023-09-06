# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
import os
import io
import importlib
import logging as log
from contextlib import redirect_stdout


skip_snippets = ["main.py", "__init__.py", "utils.py", "ie_common.py", "ov_common.py"]

def import_python_modules(directory, subdirectory=""):
    for item in os.listdir(directory):
        if item.endswith('.py') and item not in skip_snippets:
            imported_item = item[:-3]
            if subdirectory != "":
                imported_item=subdirectory + "." + imported_item
            mod = importlib.import_module(imported_item)
            try:
                with redirect_stdout(io.StringIO()) as f:
                    mod.main()
            except AttributeError as e:
                pass
            log.info(f"Snippet {item} succesfully executed.")
        
        if os.path.isdir(os.path.join(directory, item)):
            dir_path = os.path.join(directory, item)
            import_python_modules(dir_path, item)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import_python_modules(dir_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
