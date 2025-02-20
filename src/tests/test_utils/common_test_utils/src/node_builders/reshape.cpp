// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/reshape.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape) {
    if (value.get_partial_shape().same_scheme(shape)) {
        return value.get_node_shared_ptr();
    } else if (is_scalar(shape)) {
        auto value_rank = value.get_shape().size();
        AxisVector axes_vector(value_rank);
        std::iota(axes_vector.begin(), axes_vector.end(), 0);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, Shape{value_rank}, axes_vector);
        return std::make_shared<ov::op::v0::Squeeze>(value, axes);
    } else {
        auto out_pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                        Shape{shape.size()},
                                                        std::vector<int64_t>(shape.begin(), shape.end()));

        return std::make_shared<ov::op::v1::Reshape>(value, out_pattern, false);
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
