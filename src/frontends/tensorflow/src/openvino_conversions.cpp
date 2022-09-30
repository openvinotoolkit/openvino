// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_conversions.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank) {
    if (need_convert) {
        if (input_rank.is_dynamic()) {
            // TODO: use ShapeOf sub-graph to generate permutation vector
            OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                            "For conversion into the first channel format, the input rank must be static or determined "
                            "based on the operation.");
            input_rank = node.get_partial_shape().rank();
        }
        auto rank_value = input_rank.get_length();
        if (rank_value == 4) {
            node = make_transpose(node, {0, 3, 1, 2});
        } else if (rank_value == 5) {
            node = make_transpose(node, {0, 4, 1, 2, 3});
        }
    }
}

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank) {
    if (need_convert) {
        if (input_rank.is_dynamic()) {
            // TODO: use ShapeOf sub-graph to generate permutation vector
            OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                            "For conversion into the last channel format, the input rank must be static or determined "
                            "based on the operation.");
            input_rank = node.get_partial_shape().rank();
        }
        auto rank_value = input_rank.get_length();
        if (rank_value == 4) {
            node = make_transpose(node, {0, 2, 3, 1});
        } else if (rank_value == 5) {
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
