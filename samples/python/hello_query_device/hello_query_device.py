#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log
import sys

from openvino.runtime import Core


def param_to_string(metric) -> str:
    """Convert a list / tuple of parameters returned from IE to a string"""
    if isinstance(metric, (list, tuple)):
        return ', '.join([str(x) for x in metric])
    else:
        return str(metric)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core --------------------------------------------
    core = Core()

    # --------------------------- Step 2. Get metrics of available devices --------------------------------------------
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device} :')
        log.info('\tSUPPORTED_METRICS:')
        for metric in core.get_metric(device, 'SUPPORTED_METRICS'):
            if metric not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS'):
                try:
                    metric_val = core.get_metric(device, metric)
                except TypeError:
                    metric_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{metric}: {param_to_string(metric_val)}')
        log.info('')

        log.info('\tSUPPORTED_CONFIG_KEYS (default values):')
        for config_key in core.get_metric(device, 'SUPPORTED_CONFIG_KEYS'):
            try:
                config_val = core.get_config(device, config_key)
            except TypeError:
                config_val = 'UNSUPPORTED TYPE'
            log.info(f'\t\t{config_key}: {param_to_string(config_val)}')
        log.info('')

    # -----------------------------------------------------------------------------------------------------------------
    return 0


if __name__ == '__main__':
    sys.exit(main())
