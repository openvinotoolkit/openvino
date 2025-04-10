# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import builtins
import torch
from typing import Callable, Dict

originals_map = {
    'divmod': builtins.divmod
}


def patched_divmod(a, b):
    if isinstance(a, torch.Tensor):
        q = torch.div(a, b, rounding_mode='trunc')
        r = torch.remainder(a, b)
        return q, r
    return originals_map['divmod'](a, b)


patches_map = {
    'divmod': patched_divmod,
}


class BuiltinPatcher:
    def __enter__(self):
        for name, new_fn in patches_map.items():
            setattr(builtins, name, new_fn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, _ in patches_map.items():
            original_fn = originals_map[name]
            setattr(builtins, name, original_fn)
