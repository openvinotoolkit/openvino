// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

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
        return make_shared<v0::Floor>(make_shared<v1::Divide>(x, y));
    };
    return translate_binary_op(node, floordiv_fn);
}

template <typename T>
OutputVector translate_binary_op(const NodeContext& node) {
    return translate_binary_op(node, [](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
    });
}

template OutputVector translate_binary_op<v1::Add>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseAnd>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseOr>(const NodeContext& node);
template OutputVector translate_binary_op<v13::BitwiseXor>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Equal>(const NodeContext& node);
template OutputVector translate_binary_op<v1::FloorMod>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Greater>(const NodeContext& node);
template OutputVector translate_binary_op<v1::GreaterEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Less>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LessEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalAnd>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalOr>(const NodeContext& node);
template OutputVector translate_binary_op<v1::LogicalXor>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Maximum>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Minimum>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Multiply>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Mod>(const NodeContext& node);
template OutputVector translate_binary_op<v1::NotEqual>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Power>(const NodeContext& node);
template OutputVector translate_binary_op<v0::PRelu>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Divide>(const NodeContext& node);
template OutputVector translate_binary_op<v0::SquaredDifference>(const NodeContext& node);
template OutputVector translate_binary_op<v1::Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
