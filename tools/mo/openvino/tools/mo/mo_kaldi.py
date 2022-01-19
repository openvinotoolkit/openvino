#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


if __name__ == "__main__":
    from openvino.tools.mo.subprocess_main import subprocess_main
    subprocess_main(framework='kaldi')
