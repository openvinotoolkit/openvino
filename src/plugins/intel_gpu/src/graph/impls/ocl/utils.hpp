// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"

namespace cldnn {
namespace ocl {

int32_t evaluateJIT(const std::string& expression, const int32_t* shape_info_ptr);

}  // namespace ocl
}  // namespace cldnn
