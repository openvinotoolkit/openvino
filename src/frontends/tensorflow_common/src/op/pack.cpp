// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "helper_ops/complex_type_mark.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_pack_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Pack", "PACK"}, true);

    auto num_size = static_cast<int>(node.get_input_size());
    auto axis = node.get_attribute<int64_t>("axis", 0);
    auto axis_const = make_shared<v0::Constant>(element::i64, Shape{}, axis);

    OutputVector concat_inputs;
    bool has_complex_input = false;
    ov::element::Type complex_part_type;

    for (int ind = 0; ind < num_size; ++ind) {
        auto in = node.get_input(ind);
        auto complex_type_mark = as_type_ptr<ComplexTypeMark>(in.get_node_shared_ptr());

        if (complex_type_mark) {
            has_complex_input = true;
            complex_part_type = complex_type_mark->get_complex_part_type();
            in = complex_type_mark->input_value(0);
        }

        concat_inputs.push_back(make_shared<v0::Unsqueeze>(in, axis_const));
    }

    auto concat = make_shared<v0::Concat>(concat_inputs, axis);

    if (has_complex_input) {
        OutputVector extended_concat_inputs;

        for (const auto& input : concat_inputs) {
            auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
            if (complex_type_mark) {
                auto real_part = make_shared<v0::Unsqueeze>(complex_type_mark->input_value(0), axis_const);
                extended_concat_inputs.push_back(real_part);
                auto imag_part = make_shared<v0::Unsqueeze>(complex_type_mark->input_value(1), axis_const);
                extended_concat_inputs.push_back(imag_part);
            } else {
                extended_concat_inputs.push_back(input);
            }
        }

        auto complex_concat = make_shared<v0::Concat>(extended_concat_inputs, axis);
        auto complex_pack = make_shared<ComplexTypeMark>(complex_concat, complex_part_type);
        set_node_name(node.get_name(), complex_pack);
        return {complex_pack->output(0)};
    } else {
        set_node_name(node.get_name(), concat);
        return {concat};
    }
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
