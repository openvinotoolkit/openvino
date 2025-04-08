// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "openvino/op/ops.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
// clang-format on

#include <memory>

namespace ov {
namespace op {
namespace internal {

template class TRANSFORMATIONS_API op::internal::NmsStaticShapeIE<op::v8::MatrixNms>;

}  // namespace internal
}  // namespace op
}  // namespace ov
