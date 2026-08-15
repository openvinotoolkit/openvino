// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local shape layer. Minimal replacement for openvino/core/shape.hpp.

#pragma once

#include <vector>

#include "dimension.hpp"

namespace ov {

// Same storage as upstream: int64_t dims, dynamic dims are -1.
using Shape = std::vector<Dimension::value_type>;

}  // namespace ov
