// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {
namespace partitioning {
namespace traits {

bool is_tiny_shape(const ov::Shape& shape);

bool is_tiny_scalar(const std::shared_ptr<ov::Node>& node);

}  // namespace traits
}  // namespace partitioning
}  // namespace npuw
}  // namespace ov
