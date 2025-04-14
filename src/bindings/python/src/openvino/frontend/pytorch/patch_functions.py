# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import builtins
import torch
from typing import Any, Tuple, Optional, Type

originals_map = {
    "divmod": builtins.divmod
}


def patched_divmod(lhs: Any, rhs: Any) -> Tuple[Any, Any]:
    # patch only a case with torch.Tensor input type
    # to help TorchScript to trace
    # others cases are handled without change
    if isinstance(lhs, torch.Tensor):
        div_res = torch.div(lhs, rhs, rounding_mode="trunc")
        rem = torch.remainder(lhs, rhs)
        return div_res, rem
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
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[Any]
                 ) -> None:
        for name, _ in patches_map.items():
            original_fn = originals_map[name]
            setattr(builtins, name, original_fn)
