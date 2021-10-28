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

OutputVector TranslateUnaryOp(const NodeContext& op,
                              const std::function<shared_ptr<Node>(Output<Node>)>& create_unary_op) {
    auto ng_input = op.get_ng_input(0);
    auto res = create_unary_op(ng_input);
    set_node_name(op.get_name(), res);
    return {res};
}

template <typename T>
OutputVector TranslateUnaryOp(const NodeContext& node) {
    return TranslateUnaryOp(node, [](Output<Node> n) {
        return make_shared<T>(n);
    });
}

template OutputVector TranslateUnaryOp<Abs>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Acos>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Acosh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Asin>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Asinh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Atan>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Atanh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Ceiling>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Cos>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Cosh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Exp>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Floor>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Log>(const NodeContext& node);
template OutputVector TranslateUnaryOp<LogicalNot>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Negative>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Relu>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sigmoid>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sin>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sinh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Sign>(const NodeContext& node);
template OutputVector TranslateUnaryOp<SoftPlus>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Tan>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Tanh>(const NodeContext& node);
template OutputVector TranslateUnaryOp<Swish>(const NodeContext& node);

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov