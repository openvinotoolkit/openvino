// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"
#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"
#include "helper_ops/complex_type_mark.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Slice"});
    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    auto start = node.get_input(1);
    auto size = node.get_input(2);
    auto const_one = create_same_type_const_scalar<int32_t>(start, 1);
    auto const_zero = create_same_type_const_scalar<int32_t>(start, 0);
    auto stop_pos = make_shared<v1::Add>(start, size);
    Output<Node> stop_neg = make_shared<v3::ShapeOf>(input);
    stop_neg = make_shared<v1::ConvertLike>(stop_neg, size);
    auto negative_sizes_mask = make_shared<v1::Less>(size, const_zero);
    auto stop = make_shared<v1::Select>(negative_sizes_mask, stop_neg, stop_pos);
    auto start_shape = make_shared<v3::ShapeOf>(start);
    auto step = make_shared<v3::Broadcast>(const_one, start_shape);
    OutputVector slice;

    if(complex_type_mark){
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_part_type->input_value(0);
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto real = make_shared<v8::Gather>(input, gather_index_real, minus_one)->output(0);
        auto imag = make_shared<v8::Gather>(input, gather_index_imag, minus_one)->output(0);

        auto real_part = make_shared<v8::Slice>(real,start,stop,step);
        set_node_name(node.get_name(), real_part);
        auto imag_part = make_shared<v8::Slice>(imag, start, stop, step);
        set_node_name(node.get_name(), imag_part);

        OutputVector concat_inputs;
        concat_inputs.push_back(real_part);
        concat_inputs.push_back(imag_part);
        auto concat = make_shared<v0::Concat>(concat_inputs, 0);

        auto complex_slice = make_shared<ComplexTypeMark>(concat, complex_part_type);
        slice = {complex_slice->output(0)};


    }

    else{
    auto res = make_shared<v8::Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    slice = res->outputs();
    }
    return slice;
    
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov