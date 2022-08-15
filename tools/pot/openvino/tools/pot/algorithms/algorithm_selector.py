# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ..utils.registry import Registry, RegistryStorage

COMPRESSION_ALGORITHMS = Registry('QuantizationAlgos')
REGISTRY_STORAGE = RegistryStorage(globals())


def get_registry(name):
    return REGISTRY_STORAGE.get_registry(name)


def get_algorithm(name):
    if name.startswith('.') or name.endswith('.'):
        raise Exception('The algorithm name cannot start or end with "."')

    if '.' in name:
        ind = name.find('.')
        reg_name = name[:ind]
        algo_name = name[ind + 1:]
    else:
        reg_name = 'QuantizationAlgos'
        algo_name = name

    reg = get_registry(reg_name)
    return reg.get(algo_name)
