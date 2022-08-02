// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_conversions.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                        "The input rank must be static to convert to the first channel format.");
        auto rank = node.get_partial_shape().rank().get_length();
        if (rank == 4) {
            node = make_transpose(node, {0, 3, 1, 2});
        } else if (rank == 5) {
            node = make_transpose(node, {0, 4, 1, 2, 3});
        }
    }
}

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                        "The input rank must be static to convert to the last channel format.");
        auto rank = node.get_partial_shape().rank().get_length();
        if (rank == 4) {
            node = make_transpose(node, {0, 2, 3, 1});
        } else if (rank == 5) {
            node = make_transpose(node, {0, 2, 3, 4, 1});
        }
    }
}

std::shared_ptr<ov::opset8::Transpose> make_transpose(const ov::Output<ov::Node>& arg,
                                                      const ov::AxisVector& input_order) {
    auto order = std::make_shared<ov::opset8::Constant>(element::i64, Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<ov::opset8::Transpose>(arg, order);
    return transpose;
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
