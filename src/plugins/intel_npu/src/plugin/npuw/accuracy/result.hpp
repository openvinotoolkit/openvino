// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {
namespace metrics {

struct Result {
    double metric = -1;
    double threshold = -1;

    operator bool() const {
        // It is unexpected that calculated metric will be < 0.0,
        // in that case, return that tensors are unequal.
        return (metric <= threshold) && (metric >= 0.0);
}
};

}  // namespace metrics
}  // namespace npuw
}  // namespace ov
