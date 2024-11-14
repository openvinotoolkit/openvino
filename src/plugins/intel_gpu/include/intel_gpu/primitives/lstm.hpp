// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"

namespace cldnn {

/// @brief Weights orders
/// @details Specifies the order in which the weights are concatenated.
/// e.g. [i, o, f, z] : [input, output, forget, block]
/// ONNX order: iofz
/// Caffe order: ifoz
/// pyTorch order: izof
/// OV order: fizo
enum class lstm_weights_order {
    iofz,
    ifoz,
    izof,
    fizo
};

struct lstm_elt : public primitive_base<lstm_elt> {
    CLDNN_DECLARE_PRIMITIVE(lstm_elt)

    lstm_elt() : primitive_base("", {}), clip(0), input_forget(0), offset_order(lstm_weights_order::iofz), direction(0) {}

    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;

    /// @brief Constructs lstm layer.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input cell Primitive id containing cell data. Provide empty string if using lstm without cell values.
    /// @param clip Clip threshold. Provide 0 if using lstm without activations clip threshold.
    /// @param input_forget Provide 0 if using lstm without coupled input-forget gates.
    /// @param offset_order. Order of the concatenated weights, recurrent, and bias. ONNX default is iofz [input, output, forget, block].
    /// @param direction default = 0, bidirectional = 1.
    lstm_elt(const primitive_id& id,
             const input_info& input,
             const primitive_id& cell = "",
             const float clip = 0,
             const bool input_forget = 0,
             const std::vector<activation_func> activations = {activation_func::logistic,
                                                               activation_func::hyperbolic_tan,
                                                               activation_func::hyperbolic_tan},
             const std::vector<activation_additional_params> activation_params = {},
             const lstm_weights_order offset_order = lstm_weights_order::iofz,
             const uint32_t direction = 0)
        : primitive_base(id, {input}),
          cell(cell),
          clip(clip),
          input_forget(input_forget),
          activations(activations),
          activation_params(activation_params),
          offset_order(offset_order),
          direction(direction) {}

    /// @brief Primitive id containing the initial value of the cell state data.
    primitive_id cell;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip;
    /// @brief Couple the input and forget gates if input_forget is 1. Default is 0.
    bool input_forget;
    /// @brief A list of 3 activation functions for the input, output, forget, cell, and hidden.
    std::vector<activation_func> activations;
    /// @brief Optional scaling values used by some activation functions. The values are consumed in the order of activation functions.
    std::vector<activation_additional_params> activation_params;
    /// @brief Weights, recurrent weights, and biases order. [iofz] : ONNX, [ifoz] : Caffe
    lstm_weights_order offset_order;
    /// @brief direction default = 0, bidirectional = 1.
    uint32_t direction;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, clip);
        seed = hash_combine(seed, input_forget);
        seed = hash_range(seed, activations.begin(), activations.end());
        for (auto& act_param : activation_params) {
            seed = hash_combine(seed, act_param.a);
            seed = hash_combine(seed, act_param.b);
        }
        seed = hash_combine(seed, offset_order);
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, cell.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lstm_elt>(rhs);

        bool act_params_eq = activation_params.size() == rhs_casted.activation_params.size();
        for (size_t i = 0; i < activation_params.size(); ++i) {
            act_params_eq &= activation_params[i].a == rhs_casted.activation_params[i].a &&
                             activation_params[i].b == rhs_casted.activation_params[i].b;
        }

        #define cmp_fields(name) name == rhs_casted.name
        return act_params_eq &&
               cmp_fields(clip) &&
               cmp_fields(input_forget) &&
               cmp_fields(activations) &&
               cmp_fields(offset_order) &&
               cmp_fields(direction) &&
               cmp_fields(cell.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm_elt>::save(ob);
        ob << cell;
        ob << clip;
        ob << input_forget;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << direction;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm_elt>::load(ib);
        ib >> cell;
        ib >> clip;
        ib >> input_forget;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> direction;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!cell.empty())
            ret.push_back(cell);
        return ret;
    }
};

}  // namespace cldnn
