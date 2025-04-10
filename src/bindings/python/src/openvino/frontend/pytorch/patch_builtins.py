# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import builtins
import torch

originals_map = {
    "divmod": builtins.divmod
}


def patched_divmod(lhs, rhs):
    if isinstance(lhs, torch.Tensor):
        div_res = torch.div(lhs, rhs, rounding_mode="trunc")
        rem = torch.remainder(lhs, rhs)
        return div_res, rem
    return originals_map["divmod"](lhs, rhs)


patches_map = {
    "divmod": patched_divmod,
}


class BuiltinPatcher:
    def __enter__(self):
        for name, new_fn in patches_map.items():
            setattr(builtins, name, new_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, _ in patches_map.items():
            original_fn = originals_map[name]
            setattr(builtins, name, original_fn)
