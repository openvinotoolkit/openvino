// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
static void slice_pads_begin_end(const Output<Node>& paddings,
                                 shared_ptr<Node>& pads_begin,
                                 shared_ptr<Node>& pads_end,
                                 bool is_complex = false) {
    // TODO: fix IR reader to accept padding of i32 type
    auto paddings_i64 = make_shared<v0::Convert>(paddings, element::i64);
    auto axis = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto index_zero = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto index_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    pads_begin = make_shared<v8::Gather>(paddings_i64, index_zero, axis);
    pads_end = make_shared<v8::Gather>(paddings_i64, index_one, axis);

    if (is_complex) {
        // the last dimension is auxiliary and needs no padding
        auto const_zero = make_shared<v0::Constant>(element::i64, Shape{1}, 0);
        pads_begin = make_shared<v0::Concat>(OutputVector{pads_begin, const_zero}, 0);
        pads_end = make_shared<v0::Concat>(OutputVector{pads_end, const_zero}, 0);
    }
}

static OutputVector translate_pad_base_op(const NodeContext& node,
                                          Output<Node>& input,
                                          const Output<Node>& paddings,
                                          const Output<Node>& constant_value) {
    bool is_complex = false;
    element::Type complex_part_type = element::f32;
    if (auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr())) {
        is_complex = true;
        complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->get_data();
    }
    auto pad_mode = ov::op::PadMode::CONSTANT;

    // prepare pads_begin and pads_end for OpenVINO Pad
    shared_ptr<Node> pads_begin, pads_end;
    slice_pads_begin_end(paddings, pads_begin, pads_end, is_complex);

    auto pad = make_shared<v1::Pad>(input, pads_begin, pads_end, constant_value, pad_mode);
    set_node_name(node.get_name(), pad);

    if (is_complex) {
        // need to propagate ComplexTypeMark after Pad operation
        auto complex_type_mark = make_shared<ComplexTypeMark>(pad, complex_part_type);
        return {complex_type_mark};
    }
    return {pad};
}

OutputVector translate_pad_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Pad", "PAD"}, true);
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);

    auto constant_value = create_same_type_const_scalar<int32_t>(input, 0);
    if (auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr())) {
        constant_value = create_same_type_const_scalar<int32_t>(complex_type_mark->get_data(), 0);
    }

    return translate_pad_base_op(node, input, paddings, constant_value);
}

OutputVector translate_padv2_op(const NodeContext& node) {
    default_op_checks(node, 3, {"PadV2", "PADV2"}, true);
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);
    auto constant_value = node.get_input(2);
    if (auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr())) {
        input = complex_type_mark->get_data();
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();

        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto x_real = make_shared<v8::Gather>(input, gather_index_real, minus_one)->output(0);
        auto x_imag = make_shared<v8::Gather>(input, gather_index_imag, minus_one)->output(0);

        auto constant_complex_type_mark = as_type_ptr<ComplexTypeMark>(constant_value.get_node_shared_ptr());
        auto constant_input = constant_complex_type_mark->get_data();
        auto constant_value_real = make_shared<v8::Gather>(constant_input, gather_index_real, minus_one)->output(0);
        auto constant_value_imag = make_shared<v8::Gather>(constant_input, gather_index_imag, minus_one)->output(0);

        auto y_real = translate_pad_base_op(node, x_real, paddings, constant_value_real)[0];
        auto y_imag = translate_pad_base_op(node, x_imag, paddings, constant_value_imag)[0];

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(y_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(y_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);

        set_node_name(node.get_name(), concat_result);
        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type);
        return {complex_result};
    }

    return translate_pad_base_op(node, input, paddings, constant_value);
}

OutputVector translate_mirror_pad_op(const NodeContext& node) {
    default_op_checks(node, 2, {"MirrorPad", "MIRROR_PAD"});
    auto input = node.get_input(0);
    auto paddings = node.get_input(1);

    // retrieve attributes
    auto mode = node.get_attribute<std::string>("mode");
    auto pad_mode = convert_padding_mode(node, mode);

    // prepare pads_begin and pads_end for OpenVINO Pad
    shared_ptr<Node> pads_begin, pads_end;
    slice_pads_begin_end(paddings, pads_begin, pads_end);

    auto pad = make_shared<v1::Pad>(input, pads_begin, pads_end, pad_mode);
    set_node_name(node.get_name(), pad);
    return {pad};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
