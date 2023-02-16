# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    import jstyleson as json
except ImportError:
    import json
from pathlib import Path
import yaml


def read_config_from_file(path):
    path = Path(path)
    extension = path.suffix.lower()
    with path.open() as f:
        if extension in ('.yaml', '.yml'):
            return yaml.load(f, Loader=yaml.SafeLoader)
        if extension in ('.json',):
            return json.load(f)
        raise RuntimeError('Unknown file extension for the file "{}"'.format(path))


def write_config_to_file(data, path):
    path = Path(path)
    extension = path.suffix.lower()
    with path.open('w') as f:
        if extension in ('.yaml', '.yml'):
            yaml.dump(data, f)
        elif extension in ('.json',):
            json.dump(data, f)
        else:
            raise RuntimeError('Unknown file extension for the file "{}"'.format(path))
