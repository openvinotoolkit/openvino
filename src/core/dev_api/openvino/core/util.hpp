// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace util {

OPENVINO_API std::vector<int64_t> read_index_vector(const ov::Tensor& tensor);

}
}  // namespace ov
