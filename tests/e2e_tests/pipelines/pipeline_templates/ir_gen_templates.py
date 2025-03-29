# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def common_ir_generation(mo_out, precision, **kwargs):
    return ("get_ir", {"get_ovc_model": {"mo_out": mo_out,
                                         "precision": precision,
                                         "additional_args": kwargs}})


def ir_pregenerated(xml, bin=None):
    if not bin:
        bin = str(Path(xml).with_suffix(".bin"))
    return "get_ir", {"pregenerated": {"xml": xml, "bin": bin}}
