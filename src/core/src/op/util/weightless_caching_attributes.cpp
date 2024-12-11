// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"

bool ov::WeightlessCacheAttribute::is_copyable() const {
    return false;
}
