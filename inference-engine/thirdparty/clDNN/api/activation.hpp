/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief activation functions
enum class activation_func {
    none,                 // val
    logistic,             // 1/(1 + exp(-val))
    hyperbolic_tan,       // tanh(val)
    relu,                 // max(0, val)
    relu_negative_slope,  // max(0, val) + a * min(0, val)    (a is additional param)
    clamp,                // max(a, min(b, val)               (a,b are additional param)
    softrelu,             // log(1 + exp(val))
    abs,                  // abs(val)
    linear,               // a*val + b                        (a,b are additional params)
    square,               // val*val
    sqrt,                 // sqrt(val)
    elu,                  // max(0, val) + a * (exp(min(0, val) - 1) (a is additional param)
    sin,                  // sin(val)
    asin,                 // asin(val)
    sinh,                 // sinh(val)
    asinh,                // asinh(val)
    cos,                  // cos(val)
    acos,                 // acos(val)
    cosh,                 // cosh(val)
    acosh,                // acosh(val)
    log,                  // log(val)
    log2,                 // log2(val)
    exp,                  // exp(val)
    tan,                  // tan(val)
    atan,                 // atan(val)
    atanh,                // atanh(val)
    floor,                // floor(val)
    ceil,                 // ceil(val)
    negative,             // -val
    negation,             // !val
    pow,                  // pow(val, a)
    reciprocal,           // (1/val)
    erf,                  // Gauss error function
    hard_sigmoid,         // max(0, min(1, a * val + b))       (a,b are additional params)
    selu,                 // for val <= 0: b * (a * e^val - a); for val > 0: b * val (a,b are additional params)
    sign,                 // val > 0: 1; val < 0: -1; val == 0: 0
    softplus,             // ln(exp(val) + 1)
    softsign,             // (val/(1+|val|))
    swish,                // (val*sigmoid(val))
    gelu                  // (0.5*val*(1 + erf(val / sqrt(2)))
};

/// @brief activation gradient functions
enum class activation_grad_func {
    none,                 // val
    relu,                 // val * (input > 0)
    relu_negative_slope,  // val * ((input > 0) + a * (input <= 0)    (a is additional param)
};

/// @brief activation additional params
struct activation_additional_params {
    float a, b;
};

/// @brief Activation using rectified linear unit or parameterized rectified linear unit.
/// @details Can get one negative slope or negative slope per channel.
/// @par Algorithm:
///   out(i,x,y) = max(0, in(i,x,y)) + slope(i) * min(0, in(i,x,y))
/// @par Where:
///   @li out(i,x,y) : value at x, y from i-th feature map after activation.
///   @li in(i,x,y) : value at x, y from i-th feature map before activation.
///   @li slope(i) : the slope value of the i-th feature map (can be shared across channels or one slope per channel).
struct activation : public primitive_base<activation> {
    CLDNN_DECLARE_PRIMITIVE(activation)

    /// @brief Constructs Relu primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param activation_func activation function.
    /// @param additional_params additional params (slope/max_val/linear a,b).
    activation(const primitive_id& id,
               const primitive_id& input,
               activation_func activation_function,
               activation_additional_params additional_params = {0.f, 0.f},
               const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          activation_function(activation_function),
          additional_params(additional_params),
          additional_params_input("") {}

    /// @brief Constructs activation with input per feature.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param additional_params_input additional params stored on a memory.
    /// Input x dimension should be equal to input feature size (one value per channel. in case of linear is one pair per channel).
    /// All other dimensions should be 1.
    activation(const primitive_id& id,
               const primitive_id& input,
               const primitive_id& additional_params_input,
               activation_func activation_function,
               const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          activation_function(activation_function),
          additional_params({0, 0}),
          additional_params_input(additional_params_input) {}

    /// @brief activation function.
    activation_func activation_function;

    /// @brief activation additional params.
    activation_additional_params additional_params;

    /// @brief PRelu activation slope input primitive id.
    /// Input x dimension should be equal to input feature size (one slope per channel).
    /// All other dimensions should be 1.
    primitive_id additional_params_input;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        if (additional_params_input.empty())
            return {};
        return {additional_params_input};
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
