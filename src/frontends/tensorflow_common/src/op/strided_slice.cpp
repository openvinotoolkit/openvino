// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_strided_slice_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto begin = node.get_input(1);
    auto end = node.get_input(2);
    auto strides = node.get_input(3);

    auto begin_mask = node.get_attribute<int64_t>("begin_mask", 0);
    auto end_mask = node.get_attribute<int64_t>("end_mask", 0);
    auto new_axis_mask = node.get_attribute<int64_t>("new_axis_mask", 0);
    auto ellipsis_mask = node.get_attribute<int64_t>("ellipsis_mask", 0);
    auto shrink_axis_mask = node.get_attribute<int64_t>("shrink_axis_mask", 0);

    auto mask_to_vector = [](int64_t mask) {
        size_t length = sizeof(mask) * CHAR_BIT;
        vector<int64_t> vec(length, 0);
        if (mask == 0) {
            return vec;
        }
        for (size_t i = 0; i < length; ++i) {
            if (static_cast<unsigned char>(mask >> i & 0x1) == 1) {
                vec[i] = 1;
            }
        }
        return vec;
    };

    auto res = make_shared<StridedSlice>(input,
                                         begin,
                                         end,
                                         strides,
                                         mask_to_vector(begin_mask),
                                         mask_to_vector(end_mask),
                                         mask_to_vector(new_axis_mask),
                                         mask_to_vector(shrink_axis_mask),
                                         mask_to_vector(ellipsis_mask));
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
