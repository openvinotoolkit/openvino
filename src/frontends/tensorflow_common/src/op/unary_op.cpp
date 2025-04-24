// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::frontend::tensorflow;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_unary_op(const NodeContext& op,
                                const function<shared_ptr<Node>(Output<Node>)>& create_unary_op) {
    default_op_checks(op, 1, {});

    auto input = op.get_input(0);
    auto res = create_unary_op(input);
    set_node_name(op.get_name(), res);
    return {res};
}

template <typename T>
OutputVector translate_unary_op(const NodeContext& node) {
    return translate_unary_op(node, [](Output<Node> n) {
        return make_shared<T>(n);
    });
}

template OutputVector translate_unary_op<v0::Abs>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Acos>(const NodeContext& node);
template OutputVector translate_unary_op<v3::Acosh>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Asin>(const NodeContext& node);
template OutputVector translate_unary_op<v3::Asinh>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Atan>(const NodeContext& node);
template OutputVector translate_unary_op<v3::Atanh>(const NodeContext& node);
template OutputVector translate_unary_op<v13::BitwiseNot>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Ceiling>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Cos>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Cosh>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Erf>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Exp>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Floor>(const NodeContext& node);
template OutputVector translate_unary_op<v4::HSwish>(const NodeContext& node);
template OutputVector translate_unary_op<v10::IsFinite>(const NodeContext& node);
template OutputVector translate_unary_op<v10::IsInf>(const NodeContext& node);
template OutputVector translate_unary_op<v10::IsNaN>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Log>(const NodeContext& node);
template OutputVector translate_unary_op<v1::LogicalNot>(const NodeContext& node);
template OutputVector translate_unary_op<v4::Mish>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Negative>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Relu>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Sigmoid>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Sin>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Sinh>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Sign>(const NodeContext& node);
template OutputVector translate_unary_op<v4::SoftPlus>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Tan>(const NodeContext& node);
template OutputVector translate_unary_op<v0::Tanh>(const NodeContext& node);
template OutputVector translate_unary_op<v9::SoftSign>(const NodeContext& node);
template OutputVector translate_unary_op<v4::Swish>(const NodeContext& node);

OutputVector translate_selu_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Selu"});
    auto features = node.get_input(0);

    // create pre-defined constants
    auto alpha = create_same_type_const<float>(features, {1.67326324f}, Shape{1});
    auto scale = create_same_type_const<float>(features, {1.05070098f}, Shape{1});
    auto selu = make_shared<v0::Selu>(features, alpha, scale);
    set_node_name(node.get_name(), selu);
    return {selu};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
