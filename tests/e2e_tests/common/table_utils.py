# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Common table formatting/creation utils used across E2E tests framework."""
#pylint:disable=import-error
from tabulate import tabulate


def make_table(*args, **kwargs):
    """Wrapper function for `tabulate` to unify table styles across tests."""
    tablefmt = kwargs.pop('tablefmt', 'orgtbl')
    table = tabulate(*args, tablefmt=tablefmt, **kwargs)
    return table
