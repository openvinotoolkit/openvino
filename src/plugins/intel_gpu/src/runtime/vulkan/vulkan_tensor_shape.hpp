// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace cldnn::vulkan {

void validate_tensor_shape(const ov::Shape& shape, const ov::element::Type& type);

}  // namespace cldnn::vulkan
