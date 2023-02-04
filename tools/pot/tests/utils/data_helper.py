# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # pylint: disable=W0221, E0202

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def dump_intermediate_data(local_path, data):
    data = json.dumps(deepcopy(data), cls=NumpyEncoder)
    local_file = open(local_path, 'w')
    json.dump(data, local_file)


def load_json(stats_path):
    with open(stats_path) as json_file:
        return json.load(json_file)
