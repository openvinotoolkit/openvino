// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> norm_vector(const NodeContext& context,
                         Output<Node> input_tensor,
                         Output<Node> dim,
                         float p,
                         bool keep_dim) {
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
    } else if (p == 0) {
        auto input_rank = input_tensor.get_partial_shape().rank();
        PYTORCH_OP_CONVERSION_CHECK(input_rank.is_dynamic() || input_rank.get_length() == 1,
                                    "ord=0 supported only for vector norm");
        auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input_tensor));
        auto cond = context.mark_node(std::make_shared<v1::NotEqual>(input_tensor, zero));
        cond = context.mark_node(std::make_shared<v1::ConvertLike>(cond, input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceSum>(cond, dim, keep_dim));
    } else {
        auto const_p = context.mark_node(v0::Constant::create(element::f32, Shape{}, {p}));
        const_p = context.mark_node(std::make_shared<v1::ConvertLike>(const_p, input_tensor));
        auto const_p_inv = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0 / p}));
        const_p_inv = context.mark_node(std::make_shared<v1::ConvertLike>(const_p_inv, input_tensor));
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto pow = context.mark_node(std::make_shared<v1::Power>(abs, const_p));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(pow, dim, keep_dim));
        res = context.mark_node(std::make_shared<v1::Power>(sum, const_p_inv));
    }
    return res;
};

Output<Node> norm_matrix(const NodeContext& context,
                         Output<Node> input_tensor,
                         Output<Node> dim,
                         float p,
                         bool keep_dim) {
    Output<Node> res;
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto first_dim = context.mark_node(std::make_shared<v8::Gather>(dim, zero, zero));
    auto second_dim = context.mark_node(std::make_shared<v8::Gather>(dim, one, zero));
    if (p == 1) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, first_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(sum, second_dim, true));
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, second_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(sum, first_dim, true));
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, second_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(sum, first_dim, true));
    } else if (p == -1) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, first_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(sum, second_dim, true));
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p, " for matrix norm");
    }
    if (!keep_dim) {
        res = context.mark_node(std::make_shared<v0::Squeeze>(res, dim));
    }

    return res;
};

Output<Node> frobenius_norm(const NodeContext& context, Output<Node> x, Output<Node> dim, bool keep_dim) {
    auto sqr = context.mark_node(std::make_shared<v1::Multiply>(x, x));
    auto sumsqr = context.mark_node(std::make_shared<v1::ReduceSum>(sqr, dim, keep_dim));
    return context.mark_node(std::make_shared<v0::Sqrt>(sumsqr));
}
};  // namespace

OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input_tensor = context.get_input(0);
    auto p_node_type = context.get_input_type(1);
    bool keep_dim = false;
    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, input_tensor);
    } else {
        dim = context.get_input(2);
    }
    if (!context.input_is_none(3)) {
        keep_dim = context.const_input<bool>(3);
    }
    if (!context.input_is_none(4)) {
        input_tensor = apply_dtype(context, 4, input_tensor);
    }
    Output<Node> res;
    if (p_node_type.is<type::Str>()) {
        auto p_str = context.const_input<std::string>(1);
        if (p_str == "fro") {
            res = frobenius_norm(context, input_tensor, dim, keep_dim);
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
        }
    } else {
        auto p = context.const_input<float>(1);
        res = norm_vector(context, input_tensor, dim, p, keep_dim);
    }
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, res);
    }
    return {res};
};

OutputVector translate_weight_norm(const NodeContext& context) {
    // aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
    num_inputs_check(context, 3, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    Output<Node> dim;
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto rank = std::get<1>(get_shape_rank(context, x, true));
    if (context.input_is_none(2)) {
        dim = context.mark_node(std::make_shared<v0::Range>(zero, rank, one));
    } else {
        dim = get_input_as_i32(context, 2);
        auto dims_before = context.mark_node(std::make_shared<v0::Range>(zero, dim, one));
        auto dim_next = context.mark_node(std::make_shared<v1::Add>(dim, one));
        auto dims_after = context.mark_node(std::make_shared<v0::Range>(dim_next, rank, one));
        dim = context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, dims_after}, 0));
    }
    Output<Node> res;
    auto norm = context.mark_node(std::make_shared<v4::ReduceL2>(x, dim, true));
    auto y_norm = context.mark_node(std::make_shared<v1::Divide>(y, norm));
    return {context.mark_node(std::make_shared<v1::Multiply>(x, y_norm))};
};

OutputVector translate_linalg_vector_norm(const NodeContext& context) {
    // aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType?
    // dtype=None) -> Tensor
    // aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool
    // keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!):
    num_inputs_check(context, 4, 6);
    auto x = context.get_input(0);
    // ord defines the vector norm that is computed.
    auto ord = context.const_input<float>(1);
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> dim;
    Output<Node> result;
    // If dim= None, x will be flattened before the norm is computed.
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, x);
    } else {
        dim = context.get_input(2);
    }
    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.vector_norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    result = norm_vector(context, x, dim, ord, keep_dim);
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_linalg_matrix_norm(const NodeContext& context) {
    // aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2, -1], bool keepdim=False, *, ScalarType?
    // dtype=None, Tensor(a!) out) -> Tensor(a!) aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2, -1],
    // bool keepdim=False, *, ScalarType? dtype=None) aten::linalg_matrix_norm.str_ord(Tensor self, str ord="fro", int[]
    // dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None)
    num_inputs_check(context, 5, 6);
    auto x = context.get_input(0);
    // ord defines the vector norm that is computed can be string or number
    auto ord_type = context.get_input_type(1);
    auto dim = context.get_input(2);
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> result;

    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.mtrix_norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    if (ord_type.is<type::Str>()) {
        auto p_str = context.const_input<std::string>(1);
        if (p_str == "fro") {
            result = frobenius_norm(context, x, dim, keep_dim);
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
        }
    } else {
        auto p = context.const_input<float>(1);
        result = norm_matrix(context, x, dim, p, keep_dim);
    }
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_linalg_norm(const NodeContext& context) {
    // aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)
    // aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)
    // aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType?
    // dtype=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 5, 6);
    auto x = context.get_input(0);
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> result;
    Output<Node> dim;
    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    // If dim=None apply for all dimensions
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, x);
    } else {
        dim = context.get_input(2);
    }
    // default norm for matrix is frobenius norm, for vector - L2, for other ranks are not determined
    if (context.input_is_none(1)) {
        auto input_rank = x.get_partial_shape().rank();
        if (input_rank.is_static() && input_rank.get_length() == 2) {
            result = frobenius_norm(context, x, dim, keep_dim);
        } else if (input_rank.is_dynamic() || input_rank.get_length() == 1) {
            result = norm_vector(context, x, dim, 2, keep_dim);
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "linalg norm for tensor rank > 2 without ord specification unsupported");
        }
    } else {
        // ord defines the  norm that is computed can be string or number
        auto ord_type = context.get_input_type(1);
        if (ord_type.is<type::Str>()) {
            auto p_str = context.const_input<std::string>(1);
            if (p_str == "fro") {
                result = frobenius_norm(context, x, dim, keep_dim);
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
            }
        } else {
            auto p = context.const_input<float>(1);
            if (!context.input_is_none(2)) {
                auto const_dim = context.const_input<std::vector<int64_t>>(2);
                if (const_dim.size() == 2) {
                    result = norm_matrix(context, x, dim, p, keep_dim);
                } else {
                    result = norm_vector(context, x, dim, p, keep_dim);
                }
            } else {
                result = norm_vector(context, x, dim, p, keep_dim);
            }
        }
    }

    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_frobenius_norm(const NodeContext& context) {
    // aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
    // aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    bool keep_dim = context.const_input<bool>(2);
    Output<Node> dim;
    if (context.input_is_none(1)) {
        dim = get_axes_range(context, 0);

    } else {
        dim = context.get_input(1);
    }
    auto result = frobenius_norm(context, x, dim, keep_dim);
    if (!context.input_is_none(3)) {
        context.mutate_input(3, result);
    }
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
