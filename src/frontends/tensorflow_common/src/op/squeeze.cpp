// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/string_tensor_pack.hpp"
#include "openvino/op/string_tensor_unpack.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_squeeze_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Squeeze", "SQUEEZE"}, true);

    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    std::vector<int64_t> axis;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute<std::vector<int64_t>>("axis", {});
    } else {
        // check deprecated name
        axis = node.get_attribute<std::vector<int64_t>>("squeeze_dims", {});
    }
    auto axis_const = make_shared<v0::Constant>(element::i32, Shape{axis.size()}, axis);

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->get_data();

        auto input_rank = compute_subgraph_scalar_rank(input, element::i32, true);
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
        auto input_rank_minus_one = make_shared<v1::Subtract>(input_rank, const_one)->output(0);

        // adjust axis to make them non-negative
        auto axis_complex = make_shared<v1::FloorMod>(axis_const, input_rank_minus_one);

        auto squeeze = make_shared<v0::Squeeze>(input, axis_complex);
        set_node_name(node.get_name(), squeeze);
        auto squeeze_complex = make_shared<ComplexTypeMark>(squeeze, complex_part_type);
        return {squeeze_complex->output(0)};
    } else if (input.get_element_type() == element::string) {
        ov::OutputVector unpacked_input = make_shared<v15::StringTensorUnpack>(input)->outputs();
        auto begins = unpacked_input[0];
        auto ends = unpacked_input[1];
        auto chars = unpacked_input[2];

        // squeeze begins and ends by given dimensions
        begins = std::make_shared<v0::Squeeze>(begins, axis_const);
        ends = std::make_shared<v0::Squeeze>(ends, axis_const);

        ov::Output<ov::Node> string_pack_result = make_shared<v15::StringTensorPack>(begins, ends, chars);
        set_node_name(node.get_name(), string_pack_result.get_node_shared_ptr());
        return {string_pack_result};
    }

    auto squeeze = make_shared<v0::Squeeze>(input, axis_const);
    set_node_name(node.get_name(), squeeze);
    return {squeeze};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
