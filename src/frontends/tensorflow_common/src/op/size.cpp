// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

ov::OutputVector translate_size_op(const NodeContext& node) {
    // Size operation computes a number of elements in the input tensor
    default_op_checks(node, 1, {"Size"}, true);
    auto input = node.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    // retrive attribute of the output type
    auto out_type = node.get_attribute<element::Type>("out_type", element::i32);

    if (complex_type_mark) {
        input = complex_type_mark->get_data();

        // compute the input tensor size
        auto shape_of = make_shared<v3::ShapeOf>(input, out_type);
        auto axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto complex_size = make_shared<v1::ReduceProd>(shape_of, axis);

        // need to divide the size by 2 because real and imaginary parts are counted separately
        auto complex_size_divided_by_two =
            make_shared<v1::Divide>(complex_size, make_shared<v0::Constant>(element::i32, Shape{}, 2));
        set_node_name(node.get_name(), complex_size_divided_by_two);
        return {complex_size_divided_by_two->output(0)};
    }

    // introduce extra dimension in order to compute size in case of a scalar input
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    input = make_shared<v0::Unsqueeze>(input, const_zero);

    // compute the input tensor size
    auto shape_of = make_shared<v3::ShapeOf>(input, out_type);
    auto axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto size = make_shared<v1::ReduceProd>(shape_of, axis);

    set_node_name(node.get_name(), size);
    return {size};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
