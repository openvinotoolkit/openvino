// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_zeros_like_op(const NodeContext& node) {
    default_op_checks(node, 1, {"ZerosLike", "ZEROS_LIKE"});
    auto x = node.get_input(0);
    Output<Node> shape_of = make_shared<v3::ShapeOf>(x, element::i32);
    auto zero_const = create_same_type_const_scalar<int32_t>(x, 0);

    // in case of x to be scalar, we need handle it more specifically
    // since Broadcast supports only broadcasting to rank greater 0
    // we have to introduce extra dimension for input scalar case
    auto zero_int_const = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto one_int_const = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    shape_of = make_shared<v0::Concat>(OutputVector{one_int_const, shape_of}, 0);

    // create a tensor of zeros of shape with extra dimension
    Output<Node> zeros_like = make_shared<v3::Broadcast>(zero_const, shape_of);
    // remove extra dimension by squeezing
    zeros_like = make_shared<v0::Squeeze>(zeros_like, zero_int_const);

    set_node_name(node.get_name(), zeros_like.get_node_shared_ptr());
    return {zeros_like};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
