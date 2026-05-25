// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/low_precision_dequantize.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace decomposition {

ov::Output<ov::Node> low_precision_dequantize(ov::pass::NodeRegistry& reg,
                                              const ov::Output<ov::Node>& x,
                                              const ov::Output<ov::Node>& scale,
                                              const ov::Output<ov::Node>& zero_point,
                                              const ov::Output<ov::Node>& output_shape) {
    // Decomposition shape (matches ov::pass::MarkDequantization):
    //   Multiply(Subtract(Convert(x), zp), scale) [-> Reshape]
    // or, when zero_point is empty:
    //   Multiply(Convert(x), scale) [-> Reshape]
    const auto& dst_type = scale.get_element_type();
    ov::Output<ov::Node> result = reg.make<ov::op::v0::Convert>(x, dst_type);

    if (zero_point.get_node_shared_ptr()) {
        ov::Output<ov::Node> zp = zero_point;
        if (zp.get_element_type() != dst_type) {
            zp = reg.make<ov::op::v0::Convert>(zp, dst_type);
        }
        result = reg.make<ov::op::v1::Subtract>(result, zp);
    }

    result = reg.make<ov::op::v1::Multiply>(result, scale);

    if (output_shape.get_node_shared_ptr()) {
        result = reg.make<ov::op::v1::Reshape>(result, output_shape, /*special_zero=*/false);
    }

    return result;
}

}  // namespace decomposition
}  // namespace ov
