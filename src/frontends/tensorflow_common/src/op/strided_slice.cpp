// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/strided_slice.hpp"

#include <climits>

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_strided_slice_op(const NodeContext& node) {
    default_op_checks(node, 4, {"StridedSlice", "STRIDED_SLICE"}, true);
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

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    element::Type complex_part_type = element::dynamic;
    std::vector<int64_t> begin_axes;
    if (complex_type_mark) {
        complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->get_data();

        TENSORFLOW_OP_VALIDATION(node,
                                 as_type_ptr<v0::Constant>(node.get_input(1).get_node_shared_ptr()),
                                 "StridedSlice for complex values is not supported with non-constant begin");
        get_const_input(node, 1, &begin_axes);
        max_length = std::max(begin_axes.size() + 1, max_length);
    }

    begin_mask.resize(max_length, 0);
    end_mask.resize(max_length, 0);
    new_axis_mask.resize(max_length, 0);
    ellipsis_mask.resize(max_length, 0);
    shrink_axis_mask.resize(max_length, 0);

    if (complex_type_mark) {
        auto zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        begin = make_shared<v0::Concat>(OutputVector{begin, zero}, 0);
        end = make_shared<v0::Concat>(OutputVector{end, zero}, 0);
        strides = make_shared<v0::Concat>(OutputVector{strides, one}, 0);

        begin_mask[begin_axes.size()] = 1;
        end_mask[begin_axes.size()] = 1;
        new_axis_mask[begin_axes.size()] = 0;
        ellipsis_mask[begin_axes.size()] = 0;
        shrink_axis_mask[begin_axes.size()] = 0;
    }

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

    if (complex_type_mark) {
        auto complex_strided_slice = make_shared<ComplexTypeMark>(strided_slice, complex_part_type);
        return {complex_strided_slice->output(0)};
    }

    return {strided_slice};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
