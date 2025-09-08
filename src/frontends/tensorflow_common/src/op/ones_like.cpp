// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_ones_like_op(const NodeContext& node) {
    default_op_checks(node, 1, {"OnesLike"}, true);
    auto x = node.get_input(0);
    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark_x) {
        x = complex_type_mark_x->get_data();
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto x_real = make_shared<v8::Gather>(x, gather_index_real, minus_one)->output(0);
        Output<Node> shape_of_real = make_shared<v3::ShapeOf>(x_real, element::i32);

        auto one_const = create_same_type_const_scalar<int32_t>(x_real, 1);
        Output<Node> ones_like = make_shared<v3::Broadcast>(one_const, shape_of_real);

        auto zero_const = create_same_type_const_scalar<int32_t>(x_real, 0);
        Output<Node> zeros_like = make_shared<v3::Broadcast>(zero_const, shape_of_real);
        auto result = make_shared<v0::Concat>(OutputVector{ones_like, zeros_like}, -1);
        set_node_name(node.get_name(), result);
        auto ones_like_complex = make_shared<ComplexTypeMark>(result, complex_type_mark_x->get_complex_part_type());

        return {ones_like_complex};
    }

    Output<Node> shape_of = make_shared<v3::ShapeOf>(x, element::i32);
    auto one_const = create_same_type_const_scalar<int32_t>(x, 1);

    // in case of x to be scalar, we need handle it more specifically
    // since Broadcast supports only broadcasting to rank greater 0
    // we have to introduce extra dimension for input scalar case
    auto one_dim = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    shape_of = make_shared<v0::Concat>(OutputVector{one_dim, shape_of}, 0);

    // create a tensor of zeros of shape with extra dimension
    Output<Node> ones_like = make_shared<v3::Broadcast>(one_const, shape_of);
    // remove extra dimension by squeezing
    auto zero_dim_ind = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    ones_like = make_shared<v0::Squeeze>(ones_like, zero_dim_ind);
    set_node_name(node.get_name(), ones_like.get_node_shared_ptr());
    return {ones_like};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
