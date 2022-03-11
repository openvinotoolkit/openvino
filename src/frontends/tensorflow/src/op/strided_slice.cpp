// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_strided_slice_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto rank = input.get_partial_shape().rank();
    auto begin = node.get_input(1);
    auto end = node.get_input(2);
    auto strides = node.get_input(3);

    auto begin_mask = node.get_attribute<int64_t>("begin_mask");
    auto end_mask = node.get_attribute<int64_t>("end_mask");
    auto new_axis_mask = node.get_attribute<int64_t>("new_axis_mask");
    auto ellipsis_mask = node.get_attribute<int64_t>("ellipsis_mask");
    auto shrink_axis_mask = node.get_attribute<int64_t>("shrink_axis_mask");

    auto mask_to_vec = [](int64_t mask, const ov::Rank& rank) {
        auto length = sizeof(mask) * CHAR_BIT;
        if (rank.is_static() && rank.get_length() < length) {
            length = rank.get_length();
        }
        vector<int64_t> vec(length, 0);
        if (mask == 0) {
            return vec;
        }
        for (auto i = 0; i < length; ++i) {
            if (static_cast<unsigned char>(mask >> i & 0x01) == 1) {
                vec[i] = 1;
            }
        }
        return vec;
    };

    auto res = make_shared<StridedSlice>(input,
                                         begin,
                                         end,
                                         strides,
                                         mask_to_vec(begin_mask, rank),
                                         mask_to_vec(end_mask, rank),
                                         mask_to_vec(new_axis_mask, rank),
                                         mask_to_vec(shrink_axis_mask, rank),
                                         mask_to_vec(ellipsis_mask, rank));
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
