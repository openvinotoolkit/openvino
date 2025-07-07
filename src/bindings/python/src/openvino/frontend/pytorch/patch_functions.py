# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import builtins
import torch
from typing import Any, Optional

originals_map = {
    "divmod": builtins.divmod
}


def patched_divmod(lhs: Any, rhs: Any) -> tuple[Any, Any]:
    # TorchScript tracing outputs torch.Tensor for `x.shape[-1]` instead of scalar.
    # For example, it leads to TypeError issue during tracing of `divmod(x.shape[-1], 5)`
    # since builtin `divmod()` expects both scalar inputs.
    # So the patch for `divmod` is required to adjust operands to scalar type
    if isinstance(lhs, torch.Tensor):
        lhs = lhs.item()
    if isinstance(rhs, torch.Tensor):
        rhs = rhs.item()
    return originals_map["divmod"](lhs, rhs)


patches_map = {
    "divmod": patched_divmod,
}


class FunctionsPatcher:
    """Patch different Python functions including built-in routines such as divmod().

    For example, TorchScript is unable to trace divmod() with torch.Tensor input type.
    """
    def __enter__(self) -> None:
        for name, new_fn in patches_map.items():
            setattr(builtins, name, new_fn)

    def __exit__(self,
                 exc_type: Optional[type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[Any]
                 ) -> None:
        for name, _ in patches_map.items():
            original_fn = originals_map[name]
            setattr(builtins, name, original_fn)
