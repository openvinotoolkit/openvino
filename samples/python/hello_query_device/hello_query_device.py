#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log
import sys

import openvino as ov


def param_to_string(parameters) -> str:
    """Convert a list / tuple of parameters returned from OV to a string."""
    if isinstance(parameters, (list, tuple)):
        return ', '.join([str(x) for x in parameters])
    else:
        return str(parameters)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core --------------------------------------------
    core = ov.Core()

    # --------------------------- Step 2. Get metrics of available devices --------------------------------------------
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device} :')
        log.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_PROPERTIES'):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        log.info('')

    # -----------------------------------------------------------------------------------------------------------------
    return 0


if __name__ == '__main__':
    sys.exit(main())
