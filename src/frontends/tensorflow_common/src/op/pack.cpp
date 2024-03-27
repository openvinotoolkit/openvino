// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/less.hpp"

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
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(node.get_input(0).get_node_shared_ptr());
    auto axis_const = make_shared<v0::Constant>(element::i64, Shape{}, axis);

    if (complex_type_mark) {
        auto zero = create_same_type_const_scalar<int32_t>(axis_const, 0);
        auto less_than_zero = make_shared<v1::Less>(axis, zero);
        auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);

        auto axis_update = make_shared<v1::Select>(less_than_zero, const_one, zero); 
        auto new_axis = make_shared<v1::Subtract>(axis_const, axis_update);  
    } 

    OutputVector concat_inputs;
    for (int ind = 0; ind < num_size; ++ind) {
        auto in = node.get_input(ind);
        concat_inputs.push_back(make_shared<v0::Unsqueeze>(in, axis_const));
    }

    auto pack = make_shared<v0::Concat>(concat_inputs, axis);
    set_node_name(node.get_name(), pack);
    if (complex_type_mark) {
        auto complex_result = make_shared<ComplexTypeMark>(pack, complex_type_mark->get_complex_part_type());
        return {complex_result};
    }
    return {pack};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
