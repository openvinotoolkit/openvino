// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/opsets/opset13.hpp"

using namespace std;
using namespace ov::opset13;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_binary_op(const NodeContext& node,
                                 const std::function<Output<Node>(Output<Node>&, Output<Node>&)>& create_binary_op) {
    default_op_checks(node, 2, {});
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);
    auto result = create_binary_op(lhs, rhs);
    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}

OutputVector translate_floor_div_op(const NodeContext& node) {
    auto floordiv_fn = [](const Output<Node>& x, const Output<Node>& y) {
        auto out_type = x.get_element_type();
        return make_shared<Convert>(make_shared<Floor>(make_shared<Divide>(make_shared<Convert>(x, element::f64),
                                                                           make_shared<Convert>(y, element::f64))),
                                    out_type);
    };
    return translate_binary_op(node, floordiv_fn);
}

template <typename T>
OutputVector translate_binary_op(const NodeContext& node) {
    return translate_binary_op(node, [](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
    });
}

OutputVector translate_mul_op(const NodeContext& node) {
    default_op_checks(node, 2, {}, true);
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);
    auto result = make_shared<Multiply>(lhs, rhs);
    set_node_name(node.get_name(), result);

    auto complex_type_mark_lhs = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto complex_type_mark_rhs = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());
    if (complex_type_mark_lhs || complex_type_mark_rhs) {
        FRONT_END_GENERAL_CHECK(complex_type_mark_lhs != nullptr && complex_type_mark_rhs != nullptr,
                                "Mul gox complex and non-complex inputs. Inputs should be of same type.");
        lhs = complex_type_mark_lhs->input_value(0);
        rhs = complex_type_mark_rhs->input_value(0);

        element::Type complex_part_type_lhs = complex_type_mark_lhs->get_complex_part_type();
        element::Type complex_part_type_rhs = complex_type_mark_rhs->get_complex_part_type();
        FRONT_END_GENERAL_CHECK(complex_part_type_lhs == complex_part_type_rhs,
                                "Mul got complex inputs of different types. Inputs should be of same type.");

        auto gather_index_real = make_shared<Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<Constant>(element::i32, Shape{}, 1);

        auto minus_one = make_shared<Constant>(element::i32, Shape{1}, -1);

        auto lhs_real = make_shared<Gather>(lhs, gather_index_real, minus_one)->output(0);
        auto lhs_imag = make_shared<Gather>(lhs, gather_index_imag, minus_one)->output(0);
        auto rhs_real = make_shared<Gather>(rhs, gather_index_real, minus_one)->output(0);
        auto rhs_imag = make_shared<Gather>(rhs, gather_index_imag, minus_one)->output(0);

        // result_real = lhs_real * rhs_real - lhs_imag * rhs_imag
        auto result_real =
            make_shared<Subtract>(make_shared<Multiply>(lhs_real, rhs_real), make_shared<Multiply>(lhs_imag, rhs_imag));

        // result_imag = lhs_real * rhs_imag + lhs_imag * rhs_real
        auto result_imag =
            make_shared<Add>(make_shared<Multiply>(lhs_real, rhs_imag), make_shared<Multiply>(lhs_imag, rhs_real));

        auto real_unsqueeze = make_shared<Unsqueeze>(result_real, minus_one);
        auto imag_unsqueeze = make_shared<Unsqueeze>(result_imag, minus_one);

        auto concat_result = make_shared<Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type_lhs);
        return {complex_result};
    }

    return {result};
}

template OutputVector translate_binary_op<Add>(const NodeContext& node);
template OutputVector translate_binary_op<BitwiseAnd>(const NodeContext& node);
template OutputVector translate_binary_op<BitwiseOr>(const NodeContext& node);
template OutputVector translate_binary_op<BitwiseXor>(const NodeContext& node);
template OutputVector translate_binary_op<Equal>(const NodeContext& node);
template OutputVector translate_binary_op<FloorMod>(const NodeContext& node);
template OutputVector translate_binary_op<Greater>(const NodeContext& node);
template OutputVector translate_binary_op<GreaterEqual>(const NodeContext& node);
template OutputVector translate_binary_op<Less>(const NodeContext& node);
template OutputVector translate_binary_op<LessEqual>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalAnd>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalOr>(const NodeContext& node);
template OutputVector translate_binary_op<LogicalXor>(const NodeContext& node);
template OutputVector translate_binary_op<Maximum>(const NodeContext& node);
template OutputVector translate_binary_op<Minimum>(const NodeContext& node);
template OutputVector translate_binary_op<Multiply>(const NodeContext& node);
template OutputVector translate_binary_op<Mod>(const NodeContext& node);
template OutputVector translate_binary_op<NotEqual>(const NodeContext& node);
template OutputVector translate_binary_op<Power>(const NodeContext& node);
template OutputVector translate_binary_op<PRelu>(const NodeContext& node);
template OutputVector translate_binary_op<Divide>(const NodeContext& node);
template OutputVector translate_binary_op<SquaredDifference>(const NodeContext& node);
template OutputVector translate_binary_op<Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
