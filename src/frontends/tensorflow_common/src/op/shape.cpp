// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

std::shared_ptr<v8::Slice> compute_complex_shape(const ov::Output<ov::Node>& input, element::Type out_type) {
    auto shapeof = make_shared<v3::ShapeOf>(input, out_type);
    auto rank = make_shared<v3::ShapeOf>(shapeof, out_type);
    auto one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);

    auto start = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto stop = make_shared<v1::Subtract>(rank, one);
    auto step = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto axes = make_shared<v0::Constant>(element::i32, Shape{1}, 0);

    return make_shared<v8::Slice>(shapeof, start, stop, step, axes);
}

OutputVector translate_shape_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Shape", "ShapeN", "SHAPE"}, true);
    auto input_size = static_cast<int>(node.get_input_size());
    auto out_type = node.get_attribute<element::Type>("out_type", element::i32);
    auto node_name = node.get_name();

    if (input_size == 1) {
        auto input = node.get_input(0);

        auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
        if (complex_type_mark) {
            auto slice = compute_complex_shape(complex_type_mark->get_data(), out_type);
            set_node_name(node_name, slice);
            return {slice};
        } else {
            auto shapeof = make_shared<v3::ShapeOf>(input, out_type);
            set_node_name(node_name, shapeof);
            return {shapeof};
        }
    }

    OutputVector outputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        auto input = node.get_input(input_ind);

        auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
        if (complex_type_mark) {
            auto slice = compute_complex_shape(complex_type_mark->get_data(), out_type);
            slice->set_friendly_name(node_name + "_" + to_string(input_ind));
            auto shapeof_output = slice->output(0);
            set_out_name({node_name + ":" + to_string(input_ind)}, shapeof_output);
            outputs.push_back(shapeof_output);
        } else {
            auto shapeof = make_shared<v3::ShapeOf>(input, out_type);
            shapeof->set_friendly_name(node_name + "_" + to_string(input_ind));
            auto shapeof_output = shapeof->output(0);
            set_out_name({node_name + ":" + to_string(input_ind)}, shapeof_output);
            outputs.push_back(shapeof_output);
        }
    }

    return outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
