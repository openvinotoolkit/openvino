#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def create_fem():
    fem = None
    try:
        from ngraph import FrontEndManager # pylint: disable=no-name-in-module,import-error
        fem = FrontEndManager()
    except Exception:
        print("nGraph FrontEndManager is not initialized")
        pass
    return fem


if __name__ == "__main__":
    if not create_fem():
        exit(1)
