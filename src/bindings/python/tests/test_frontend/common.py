# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import glob
import re
import os


def get_builtin_extensions_path():
    base_path = Path(__file__).parent.parent.parent.parent.parent
    # ECE001 Expression is too complex if it's in a single line
    base_path = base_path.parent
    paths = glob.glob(os.path.join(base_path, "bin", "*", "*", "*test_builtin_extensions*"))
    for path in paths:
        if re.search(r"(lib)?test_builtin_extensions.?\.(dll|so)", path):
            return path
    raise RuntimeError("Unable to find test_builtin_extensions")
