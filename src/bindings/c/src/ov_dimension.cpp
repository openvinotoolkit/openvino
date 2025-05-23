// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_dimension.h"

#include "common.h"

bool ov_dimension_is_dynamic(const ov_dimension_t dim) {
    if (dim.min == dim.max && dim.max > 0)
        return false;
    return true;
}
