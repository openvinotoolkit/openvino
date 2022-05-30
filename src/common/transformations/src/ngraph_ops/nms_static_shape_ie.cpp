// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "ngraph/ops.hpp"
#include "ngraph_ops/nms_static_shape_ie.hpp"
// clang-format on

#include <memory>

namespace ngraph {
namespace op {
namespace internal {

template class TRANSFORMATIONS_API op::internal::NmsStaticShapeIE<op::v8::MatrixNms>;

}  // namespace internal
}  // namespace op
}  // namespace ngraph
