# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

sys.path.insert(0, Path(__file__).resolve().parent.parent.parent.as_posix()) # pylint: disable=C0413
from openvino.tools.pot.utils.config_reader import read_config_from_file, write_config_to_file

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

data = read_config_from_file(src)
write_config_to_file(data, dst)
