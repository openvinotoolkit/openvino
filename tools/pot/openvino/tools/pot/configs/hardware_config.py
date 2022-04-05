# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    import jstyleson as json
except ImportError:
    import json

from collections import OrderedDict
from addict import Dict
from .utils import product_dict


class HardwareConfig(list):
    def get(self, op_type, attr):
        def match_attrs(op_config, attr):
            for attr_name, attr_value in attr.items():
                if attr_name in op_config:
                    equal = attr_value in op_config[attr_name] if isinstance(op_config[attr_name], list) \
                        else attr_value == op_config[attr_name]
                    if not equal:
                        return False
            return True

        config = None
        for op_config in self:
            if op_config.type == op_type and match_attrs(op_config, attr):
                if config is not None:
                    raise RuntimeError('Several hardware configs were defined for operation with type {}, attr {}\n '
                                       'Specify the operation uniquely'.format(op_type, attr))
                config = op_config

        return config

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            json_config = json.load(f, object_pairs_hook=OrderedDict)
            hw_config = cls()
            hw_config.append(Dict(('target_device', json_config['target_device'])))

            configs = {}
            for algorithm_name, algorithm_config in json_config.get('config', {}).items():
                configs[algorithm_name] = {}
                for config_name, config in algorithm_config.items():
                    for key, val in config.items():
                        if not isinstance(val, list):
                            config[key] = [val]

                    configs[algorithm_name][config_name] = list(product_dict(config))

            for op_config in json_config.get('operations', []):
                for algorithm_name in op_config:
                    if algorithm_name not in configs:
                        continue
                    tmp_config = {}
                    for name, algorithm_config in op_config[algorithm_name].items():
                        if not isinstance(algorithm_config, list):
                            algorithm_config = [algorithm_config]

                        tmp_config[name] = []
                        for config_item in algorithm_config:
                            if isinstance(config_item, str):
                                tmp_config[name].extend(configs[algorithm_name][config_item])
                            else:
                                for key, val in config_item.items():
                                    if not isinstance(val, list):
                                        config_item[key] = [val]

                                tmp_config[name].extend(list(product_dict(config_item)))

                    op_config[algorithm_name] = tmp_config

                hw_config.append(Dict(op_config))

            return hw_config
