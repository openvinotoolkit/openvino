#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import enum


class InputSourceFileType(enum.IntEnum):
    image = 0
    bin = 1


def get_available_input_source_type_names():
    return [str_name for str_name, _ in InputSourceFileType.__members__.items()]
