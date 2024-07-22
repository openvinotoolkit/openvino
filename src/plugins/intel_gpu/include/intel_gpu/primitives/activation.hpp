// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"

namespace cldnn {

/// @brief activation functions
enum class activation_func {
    none,                      // val
    logistic,                  // 1/(1 + exp(-val))
    hyperbolic_tan,            // tanh(val)
    relu,                      // max(0, val)
    relu_negative_slope,       // max(0, val) + a * min(0, val)    (a is additional param)
    clamp,                     // max(a, min(b, val)               (a,b are additional param)
    softrelu,                  // log(1 + exp(val))
    abs,                       // abs(val)
    linear,                    // a*val + b                        (a,b are additional params)
    square,                    // val*val
    sqrt,                      // sqrt(val)
    elu,                       // max(0, val) + a * (exp(min(0, val) - 1) (a is additional param)
    sin,                       // sin(val)
    asin,                      // asin(val)
    sinh,                      // sinh(val)
    asinh,                     // asinh(val)
    cos,                       // cos(val)
    acos,                      // acos(val)
    cosh,                      // cosh(val)
    acosh,                     // acosh(val)
    log,                       // log(val)
    log2,                      // log2(val)
    exp,                       // exp(val)
    tan,                       // tan(val)
    atan,                      // atan(val)
    atanh,                     // atanh(val)
    floor,                     // floor(val)
    ceil,                      // ceil(val)
    negative,                  // -val
    negation,                  // !val
    pow,                       // pow(val, a)
    reciprocal,                // (1/val)
    erf,                       // Gauss error function
    hard_sigmoid,              // max(0, min(1, a * val + b))       (a,b are additional params)
    hsigmoid,                  // min(max(val + 3, 0), 6) / 6
    selu,                      // for val <= 0: b * (a * e^val - a); for val > 0: b * val (a,b are additional params)
    sign,                      // val > 0: 1; val < 0: -1; val == 0: 0
    softplus,                  // ln(exp(val) + 1)
    softsign,                  // (val/(1+|val|))
    swish,                     // (val*sigmoid(val))
    hswish,                    // val * min(max(0, val + 3), 6) / 6
    mish,                      // val*tanh(ln(1 + exp(val)))
    gelu,                      // (0.5*val*(1 + erf(val / sqrt(2)))
    gelu_tanh,                 // x⋅0.5⋅(1+tanh[sqrt(2/pi)*val(1 + 0.044715⋅val^2)])
    round_half_to_even,        // round halfs to the nearest even integer
    round_half_away_from_zero  // round the number so it's further away from zero
};

/// @brief activation additional params
struct activation_additional_params {
    float a, b;

    void save(BinaryOutputBuffer& ob) const { ob << a << b; }
    void load(BinaryInputBuffer& ib) { ib >> a >> b; }
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

    activation() : primitive_base("", {}),
                   activation_function(activation_func::none),
                   additional_params({0.f, 0.f}) {}

    /// @brief Constructs Relu primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param activation_func activation function.
    /// @param additional_params additional params (slope/max_val/linear a,b).
    activation(const primitive_id& id,
               const input_info& input,
               activation_func activation_function,
               activation_additional_params additional_params = {0.f, 0.f})
        : primitive_base(id, {input}),
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
               const input_info& input,
               const primitive_id& additional_params_input,
               activation_func activation_function)
        : primitive_base(id, {input}),
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

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, activation_function);
        seed = hash_combine(seed, additional_params.a);
        seed = hash_combine(seed, additional_params.b);
        seed = hash_combine(seed, additional_params_input.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const activation>(rhs);

        return activation_function == rhs_casted.activation_function &&
               additional_params.a == rhs_casted.additional_params.a &&
               additional_params.b == rhs_casted.additional_params.b &&
               additional_params_input.empty() == rhs_casted.additional_params_input.empty();
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<activation>::save(ob);
        ob << activation_function;
        ob << additional_params;
        ob << additional_params_input;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<activation>::load(ib);
        ib >> activation_function;
        ib >> additional_params;
        ib >> additional_params_input;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        if (additional_params_input.empty())
            return {};
        return {additional_params_input};
    }
};
}  // namespace cldnn
