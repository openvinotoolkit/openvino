// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_unary_op(const NodeContext& op,
                                const std::function<shared_ptr<Node>(Output<Node>)>& create_unary_op) {
    auto ng_input = op.get_input(0);
    auto res = create_unary_op(ng_input);
    set_node_name(op.get_name(), res);
    return {res};
}

template <typename T>
OutputVector translate_unary_op(const NodeContext& node) {
    return translate_unary_op(node, [](Output<Node> n) {
        return make_shared<T>(n);
    });
}

template OutputVector translate_unary_op<Abs>(const NodeContext& node);
template OutputVector translate_unary_op<Acos>(const NodeContext& node);
template OutputVector translate_unary_op<Acosh>(const NodeContext& node);
template OutputVector translate_unary_op<Asin>(const NodeContext& node);
template OutputVector translate_unary_op<Asinh>(const NodeContext& node);
template OutputVector translate_unary_op<Atan>(const NodeContext& node);
template OutputVector translate_unary_op<Atanh>(const NodeContext& node);
template OutputVector translate_unary_op<Ceiling>(const NodeContext& node);
template OutputVector translate_unary_op<Cos>(const NodeContext& node);
template OutputVector translate_unary_op<Cosh>(const NodeContext& node);
template OutputVector translate_unary_op<Erf>(const NodeContext& node);
template OutputVector translate_unary_op<Exp>(const NodeContext& node);
template OutputVector translate_unary_op<Floor>(const NodeContext& node);
template OutputVector translate_unary_op<Log>(const NodeContext& node);
template OutputVector translate_unary_op<LogicalNot>(const NodeContext& node);
template OutputVector translate_unary_op<Mish>(const NodeContext& node);
template OutputVector translate_unary_op<Negative>(const NodeContext& node);
template OutputVector translate_unary_op<Relu>(const NodeContext& node);
template OutputVector translate_unary_op<Sigmoid>(const NodeContext& node);
template OutputVector translate_unary_op<Sin>(const NodeContext& node);
template OutputVector translate_unary_op<Sinh>(const NodeContext& node);
template OutputVector translate_unary_op<Sign>(const NodeContext& node);
template OutputVector translate_unary_op<SoftPlus>(const NodeContext& node);
template OutputVector translate_unary_op<Tan>(const NodeContext& node);
template OutputVector translate_unary_op<Tanh>(const NodeContext& node);
template OutputVector translate_unary_op<opset9::SoftSign>(const NodeContext& node);
template OutputVector translate_unary_op<Swish>(const NodeContext& node);

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov