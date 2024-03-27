# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log


def factor_update(factor: float, real_factor: list, in_shape: list, out_shape: list, name: str):
    """ Updates factor value for layers related to image resizing such as Resample and Interp. """
    if factor is None:
        if real_factor[0] != real_factor[1]:
            log.warning(
                'Cannot deduce a single zoom factor for both height and widths for node {}: [{},{}]/[{},{}] = [{},{}]. '
                'This model will not reshape in IE.'.format(
                    name,
                    out_shape[0],
                    out_shape[1],
                    in_shape[0],
                    in_shape[1],
                    real_factor[0],
                    real_factor[1]
                )
            )
        else:
            factor = real_factor[0]
    return factor
