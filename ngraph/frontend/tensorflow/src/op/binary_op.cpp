// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateBinaryOp(const NodeContext& node,
                               const std::function<Output<Node>(Output<Node>&, Output<Node>&)>& create_binary_op) {
    auto ng_lhs = node.get_ng_input(0);
    auto ng_rhs = node.get_ng_input(1);
    auto ng_node = create_binary_op(ng_lhs, ng_rhs);
    set_node_name(node.get_name(), ng_node.get_node_shared_ptr());
    return {ng_node};
}

OutputVector TranslateFloorDivOp(const NodeContext& node) {
    auto floordiv_fn = [](const Output<Node>& x, const Output<Node>& y) {
        return make_shared<Floor>(make_shared<Divide>(x, y));
    };
    return TranslateBinaryOp(node, floordiv_fn);
}

template <typename T>
OutputVector TranslateBinaryOp(const NodeContext& node) {
    return TranslateBinaryOp(node, [](Output<Node>& ng_lhs, Output<Node>& ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
    });
}

template OutputVector TranslateBinaryOp<Add>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Equal>(const NodeContext& node);
template OutputVector TranslateBinaryOp<FloorMod>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Greater>(const NodeContext& node);
template OutputVector TranslateBinaryOp<GreaterEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Less>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LessEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LogicalAnd>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LogicalOr>(const NodeContext& node);
template OutputVector TranslateBinaryOp<LogicalXor>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Maximum>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Minimum>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Multiply>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Mod>(const NodeContext& node);
template OutputVector TranslateBinaryOp<NotEqual>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Power>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Divide>(const NodeContext& node);
template OutputVector TranslateBinaryOp<SquaredDifference>(const NodeContext& node);
template OutputVector TranslateBinaryOp<Subtract>(const NodeContext& node);

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
