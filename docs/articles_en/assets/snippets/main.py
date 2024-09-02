# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import sys
import os
import io
import importlib
from contextlib import redirect_stdout, redirect_stderr


skip_snippets = ["main.py", "__init__.py", "utils.py", "ov_common.py", "ov_stateful_models_intro.py"]

def import_python_modules(directory, subdirectory=""):
    for item in os.listdir(directory):
        if item.endswith('.py') and item not in skip_snippets:
            imported_item = item[:-3]
            if subdirectory != "":
                imported_item=subdirectory + "." + imported_item
            print(f"Snippet {item} is executing...")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                mod = importlib.import_module(imported_item)

            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    mod.main()
            except AttributeError as e:
                pass

            print(f"Snippet {item} succesfully executed.")


        if os.path.isdir(os.path.join(directory, item)):
            dir_path = os.path.join(directory, item)
            import_python_modules(dir_path, item)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import_python_modules(dir_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
