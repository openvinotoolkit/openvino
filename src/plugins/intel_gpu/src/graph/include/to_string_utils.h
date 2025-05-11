// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/activation.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "program_node.h"
#endif


#include <string>
#include <sstream>
#include <memory>

namespace cldnn {

inline std::string bool_to_str(bool cond) { return cond ? "true" : "false"; }

inline std::string dt_to_str(data_types dt) {
    return ov::element::Type(dt).to_string();
}

inline std::string fmt_to_str(format fmt) {
    return fmt.to_string();
}

inline std::string type_to_str(std::shared_ptr<const primitive> primitive) { return primitive->type_string(); }

inline std::string activation_type_to_str(activation_func activation) {
    switch (activation) {
    case activation_func::none: return "none";
    case activation_func::logistic: return "logistic";
    case activation_func::hyperbolic_tan: return "hyperbolic_tan";
    case activation_func::relu: return "relu";
    case activation_func::relu_negative_slope: return "relu_negative_slope";
    case activation_func::clamp: return "clamp";
    case activation_func::softrelu: return "softrelu";
    case activation_func::abs: return "abs";
    case activation_func::linear: return "linear";
    case activation_func::square: return "square";
    case activation_func::sqrt: return "sqrt";
    case activation_func::elu: return "elu";
    case activation_func::sin: return "sin";
    case activation_func::asin: return "asin";
    case activation_func::sinh: return "sinh";
    case activation_func::asinh: return "asinh";
    case activation_func::cos: return "cos";
    case activation_func::acos: return "acos";
    case activation_func::cosh: return "cosh";
    case activation_func::acosh: return "acosh";
    case activation_func::log: return "log";
    case activation_func::log2: return "log2";
    case activation_func::exp: return "exp";
    case activation_func::tan: return "tan";
    case activation_func::atan: return "atan";
    case activation_func::atanh: return "atanh";
    case activation_func::floor: return "floor";
    case activation_func::ceil: return "ceil";
    case activation_func::negative: return "negative";
    case activation_func::negation: return "negation";
    case activation_func::pow: return "pow";
    case activation_func::reciprocal: return "reciprocal";
    case activation_func::erf: return "erf";
    case activation_func::hard_sigmoid: return "hard_sigmoid";
    case activation_func::hsigmoid: return "hsigmoid";
    case activation_func::selu: return "selu";
    case activation_func::sign: return "sign";
    case activation_func::softplus: return "softplus";
    case activation_func::softsign: return "softsign";
    case activation_func::swish: return "swish";
    case activation_func::hswish: return "hswish";
    case activation_func::mish: return "mish";
    case activation_func::gelu: return "gelu";
    case activation_func::gelu_tanh: return "gelu_tanh";
    case activation_func::round_half_to_even: return "round_half_to_even";
    case activation_func::round_half_away_from_zero: return "round_half_away_from_zero";
    default: return "unknown activation";
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
inline std::string onednn_post_op_type_to_str(onednn_post_op_type type) {
    switch (type) {
    case onednn_post_op_type::eltwise_act: return "eltwise_act";
    case onednn_post_op_type::eltwise_clip: return "eltwise_clip";
    case onednn_post_op_type::eltwise_linear: return "eltwise_linear";
    case onednn_post_op_type::eltwise_round: return "eltwise_round";
    case onednn_post_op_type::eltwise_hardsigmoid: return "eltwise_hardsigmoid";
    case onednn_post_op_type::binary_mul: return "binary_mul";
    case onednn_post_op_type::binary_add: return "binary_add";
    case onednn_post_op_type::binary_sub: return "binary_add";
    case onednn_post_op_type::binary_max: return "binary_max";
    case onednn_post_op_type::binary_min: return "binary_min";
    case onednn_post_op_type::binary_relu: return "binary_relu";
    case onednn_post_op_type::scale: return "scale";
    case onednn_post_op_type::sum: return "sum";
    case onednn_post_op_type::optimized: return "optimized";
    case onednn_post_op_type::optimized_eltwise_act: return "optimized_eltwise_act";
    case onednn_post_op_type::optimized_eltwise_linear: return "optimized_eltwise_linear";
    case onednn_post_op_type::optimized_eltwise_clip: return "optimized_eltwise_clip";
    case onednn_post_op_type::optimized_eltwise_round: return "optimized_eltwise_round";
    case onednn_post_op_type::optimized_sum: return "optimized_sum";
    default: return "unknown";
    }
}
#endif

}  // namespace cldnn
