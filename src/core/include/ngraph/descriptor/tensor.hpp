// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/descriptor/tensor.hpp"

namespace ngraph {
namespace descriptor {
/// \brief Compile-time descriptor of a first-class value that is a tensor.
using ov::descriptor::Tensor;
using TensorLabel = std::vector<size_t>;
}  // namespace descriptor
}  // namespace ngraph
