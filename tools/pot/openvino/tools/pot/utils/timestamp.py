# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime


def get_timestamp():
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')


def get_timestamp_precise():
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S.%f')


def get_timestamp_short():
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%H-%M-%S')
