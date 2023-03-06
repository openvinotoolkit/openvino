// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> norm(const NodeContext& context, Output<Node> input_tensor, Output<Node> dim, float p, bool keep_dim) {
    Output<Node> res;
    if (p == 1) {
        res = context.mark_node(std::make_shared<v4::ReduceL1>(input_tensor, dim, keep_dim));
    } else if (p == 2) {
        res = context.mark_node(std::make_shared<v4::ReduceL2>(input_tensor, dim, keep_dim));
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(abs, dim, keep_dim));
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(abs, dim, keep_dim));
    } else {
        auto const_p = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {p}));
        const_p = context.mark_node(std::make_shared<v1::ConvertLike>(const_p, input_tensor));
        auto const_p_inv = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1.0 / p}));
        const_p_inv = context.mark_node(std::make_shared<v1::ConvertLike>(const_p_inv, input_tensor));
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto pow = context.mark_node(std::make_shared<v1::Power>(abs, const_p));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(pow, dim, keep_dim));
        res = context.mark_node(std::make_shared<v1::Power>(sum, const_p_inv));
    }
    return res;
};
};  // namespace

OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto input_tensor = context.get_input(0);
    auto p = context.const_input<float>(1);
    auto dim = context.get_input(2);
    auto keep_dim = context.const_input<bool>(3);
    auto res = norm(context, input_tensor, dim, p, keep_dim);
    return {res};
};

OutputVector translate_linalg_vector_norm(const NodeContext& context) {
    // aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType?
    // dtype=None) -> Tensor
    // aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool
    // keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!):
    num_inputs_check(context, 5, 6);
    auto x = context.get_input(0);
    // ord defines the vector norm that is computed.
    auto ord = context.const_input<float>(1);
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> dim;
    Output<Node> result;
    // If dim= None, x will be flattened before the norm is computed.
    if (context.input_is_none(2)) {
        keep_dim = false;
        auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        x = context.mark_node(std::make_shared<v1::Reshape>(x, minus_one, false));
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    } else {
        dim = context.get_input(2);
    }
    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.vector_norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(4).get_node_shared_ptr())) {
            auto dtype = convert_dtype(context.const_input<int64_t>(4));
            x = context.mark_node(std::make_shared<v0::Convert>(x, dtype));
        } else if (const auto& fw_node = cast_fw_node(context.get_input(4).get_node_shared_ptr(), "prim::dtype")) {
            auto out_tensor = fw_node->input_value(0);
            x = context.mark_node(std::make_shared<v1::ConvertLike>(x, out_tensor));
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    // sum(x != 0)
    if (ord == 0) {
        auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));
        auto cond = context.mark_node(std::make_shared<v1::NotEqual>(x, zero));
        cond = context.mark_node(std::make_shared<v1::ConvertLike>(cond, x));
        result = context.mark_node(std::make_shared<v1::ReduceSum>(cond, dim, keep_dim));
    } else {
        result = norm(context, x, dim, ord, keep_dim);
    }
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov