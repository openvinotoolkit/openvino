// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "ngraph_ops/nms_static_shape_ie.hpp"

namespace ngraph {
namespace op {
namespace internal {

std::shared_ptr<const NmsStaticShapeIE<op::v8::MulticlassNms>> CastMulticlassNms(const std::shared_ptr<ngraph::Node> &op) {
    return std::dynamic_pointer_cast<const NmsStaticShapeIE<op::v8::MulticlassNms>>(op);
}

std::shared_ptr<const NmsStaticShapeIE<op::v8::MatrixNms>> CastMatrixNms(const std::shared_ptr<ngraph::Node> &op) {
    return std::dynamic_pointer_cast<const NmsStaticShapeIE<op::v8::MatrixNms>>(op);
}
}  // namespace internal
}  // namespace op
}  // namespace ngraph
