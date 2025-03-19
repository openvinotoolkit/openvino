// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

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
    if (axis < 0 && as_type_ptr<ComplexTypeMark>(node.get_input(0).get_node_shared_ptr())) {
        // need to account auxiliary dimension for real and imaginary parts
        axis -= 1;
    }

    auto axis_const = make_shared<v0::Constant>(element::i64, Shape{}, axis);

    OutputVector concat_inputs;
    bool has_complex_input = false;
    element::Type complex_part_type;

    for (int ind = 0; ind < num_size; ++ind) {
        auto in = node.get_input(ind);
        auto complex_type_mark = as_type_ptr<ComplexTypeMark>(in.get_node_shared_ptr());
        if (complex_type_mark) {
            has_complex_input = true;
            complex_part_type = complex_type_mark->get_complex_part_type();
            in = complex_type_mark->get_data();
        }
        concat_inputs.push_back(make_shared<v0::Unsqueeze>(in, axis_const));
    }

    auto pack = make_shared<v0::Concat>(concat_inputs, axis);
    set_node_name(node.get_name(), pack);

    if (has_complex_input) {
        auto complex_pack = make_shared<ComplexTypeMark>(pack, complex_part_type);
        return {complex_pack->output(0)};
    }

    return {pack};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
