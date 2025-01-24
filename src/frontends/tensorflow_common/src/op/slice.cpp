// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Slice", "SLICE"}, true);
    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    auto start = node.get_input(1);
    auto size = node.get_input(2);

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->input_value(0);

        // Separate real and imaginary parts using Gather
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto real = make_shared<v8::Gather>(input, gather_index_real, minus_one)->output(0);
        auto imag = make_shared<v8::Gather>(input, gather_index_imag, minus_one)->output(0);

        // Auxiliary constants
        auto const_one = create_same_type_const_scalar<int32_t>(start, 1);
        auto const_zero = create_same_type_const_scalar<int32_t>(start, 0);

        // compute stop values in case non-negative sizes
        auto stop_pos = make_shared<v1::Add>(start, size);

        // compute stop values in case negative sizes
        // since TensorFlow supports only -1 among negative sizes
        // assign stop values to the data shape
        Output<Node> stop_neg = make_shared<v3::ShapeOf>(real);
        stop_neg = make_shared<v1::ConvertLike>(stop_neg, size);

        // select the correct stop value based on a sign of size value
        auto negative_sizes_mask = make_shared<v1::Less>(size, const_zero);
        // TODO: investigate if we can simplify and replace Select with FloorMod operation
        // like FloorMod(size, input_shape)
        auto stop = make_shared<v1::Select>(negative_sizes_mask, stop_neg, stop_pos);

        // broadcast step value
        auto start_shape = make_shared<v3::ShapeOf>(start);
        auto step = make_shared<v3::Broadcast>(const_one, start_shape);

        // Perform slicing on real and imaginary parts
        auto real_part = make_shared<v8::Slice>(real, start, stop, step);
        set_node_name(node.get_name(), real_part);
        auto imag_part = make_shared<v8::Slice>(imag, start, stop, step);
        set_node_name(node.get_name(), imag_part);

        // Concatenate real and imaginary parts to form complex slice
        OutputVector concat_inputs;
        auto real_part_unsqueeze = make_shared<v0::Unsqueeze>(real_part, minus_one);
        auto imag_part_unsqueeze = make_shared<v0::Unsqueeze>(imag_part, minus_one);
        concat_inputs.push_back(real_part_unsqueeze);
        concat_inputs.push_back(imag_part_unsqueeze);
        auto concat = make_shared<v0::Concat>(concat_inputs, -1);
        auto complex_slice = make_shared<ComplexTypeMark>(concat, complex_part_type);
        set_node_name(node.get_name(), complex_slice);
        return complex_slice->outputs();
    }

    // create auxiliary constants
    auto const_one = create_same_type_const_scalar<int32_t>(start, 1);
    auto const_zero = create_same_type_const_scalar<int32_t>(start, 0);

    // compute stop values in case non-negative sizes
    auto stop_pos = make_shared<v1::Add>(start, size);

    // compute stop values in case negative sizes
    // since TensorFlow supports only -1 among negative sizes
    // assign stop values to the data shape
    Output<Node> stop_neg = make_shared<v3::ShapeOf>(input);
    stop_neg = make_shared<v1::ConvertLike>(stop_neg, size);

    // select the correct stop value based on a sign of size value
    auto negative_sizes_mask = make_shared<v1::Less>(size, const_zero);
    // TODO: investigate if we can simplify and replace Select with FloorMod operation
    // like FloorMod(size, input_shape)
    auto stop = make_shared<v1::Select>(negative_sizes_mask, stop_neg, stop_pos);

    // broadcast step value
    auto start_shape = make_shared<v3::ShapeOf>(start);
    auto step = make_shared<v3::Broadcast>(const_one, start_shape);

    auto res = make_shared<v8::Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
