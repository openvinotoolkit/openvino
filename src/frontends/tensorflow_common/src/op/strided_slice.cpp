// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/strided_slice.hpp"

#include <climits>

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_strided_slice_op(const NodeContext& node) {
    default_op_checks(node, 4, {"StridedSlice", "STRIDED_SLICE"});
    auto input = node.get_input(0);
    auto begin = node.get_input(1);
    auto end = node.get_input(2);
    auto strides = node.get_input(3);

    auto mask_to_vector = [](int64_t mask) {
        if (mask == 0) {
            return vector<int64_t>{};
        }
        size_t max_length = sizeof(mask) * CHAR_BIT;
        vector<int64_t> vec;
        vec.reserve(max_length);
        for (size_t i = 0; i < max_length; ++i) {
            if (((mask >> i) & 0x1) == 1) {
                // resize the vector by appending with required number of zeros
                vec.resize(i + 1, 0);
                vec[i] = 1;
            }
        }
        return vec;
    };

    // retrieve attributes for StridedSlice operation
    auto begin_mask = mask_to_vector(node.get_attribute<int64_t>("begin_mask", 0));
    auto end_mask = mask_to_vector(node.get_attribute<int64_t>("end_mask", 0));
    auto new_axis_mask = mask_to_vector(node.get_attribute<int64_t>("new_axis_mask", 0));
    auto ellipsis_mask = mask_to_vector(node.get_attribute<int64_t>("ellipsis_mask", 0));
    auto shrink_axis_mask = mask_to_vector(node.get_attribute<int64_t>("shrink_axis_mask", 0));

    // the masks can be of different length and we need to align them by the maximum length
    size_t max_length = std::max(
        {begin_mask.size(), end_mask.size(), new_axis_mask.size(), ellipsis_mask.size(), shrink_axis_mask.size()});
    begin_mask.resize(max_length, 0);
    end_mask.resize(max_length, 0);
    new_axis_mask.resize(max_length, 0);
    ellipsis_mask.resize(max_length, 0);
    shrink_axis_mask.resize(max_length, 0);

    auto strided_slice = make_shared<v1::StridedSlice>(input,
                                                       begin,
                                                       end,
                                                       strides,
                                                       begin_mask,
                                                       end_mask,
                                                       new_axis_mask,
                                                       shrink_axis_mask,
                                                       ellipsis_mask);
    set_node_name(node.get_name(), strided_slice);
    return {strided_slice};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
