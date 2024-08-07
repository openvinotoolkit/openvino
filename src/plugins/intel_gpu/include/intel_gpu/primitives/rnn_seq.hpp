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

struct rnn_seq : public primitive_base<rnn_seq> {
    CLDNN_DECLARE_PRIMITIVE(rnn_seq)

    rnn_seq() : primitive_base("", {}), clip(0), offset_order(lstm_weights_order::iofz), direction(0) {}

    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;

    /// @brief Constructs lstm seq layer.
    /// @param id This primitive id.
    /// @param input input x primitive id.
    /// @param input input initial_hidden_state
    /// @param input input initial_cell_state //when no initial_cell_state please add default input_info
    /// @param input input sequence_lenghts
    /// @param input input W
    /// @param input input R
    /// @param input input B
    /// @param input out1_prim_id - primitive for second output due to legacy
    /// @param input out2_prim_id - primitive for third output
    /// @param clip Clip threshold. Provide 0 if using lstm without activations clip threshold.
    /// @param offset_order. Order of the concatenated weights, recurrent, and bias. ONNX default is iofz [input, output, forget, block].
    /// @param direction default = 0, bidirectional = 1.
    rnn_seq(const primitive_id& id,
             const input_info& x,
             const input_info& initial_hidden_state,
             const input_info& initial_cell_state,
             const input_info& seq_lenghts,
             const input_info& W,
             const input_info& R,
             const input_info& B,
             const primitive_id& out1_prim_id = "",
             const primitive_id& out2_prim_id = "",
             const float clip = 0,
             const std::vector<activation_func> activations = {activation_func::logistic,
                                                               activation_func::hyperbolic_tan,
                                                               activation_func::hyperbolic_tan},
             const std::vector<activation_additional_params> activation_params = {},
             const lstm_weights_order offset_order = lstm_weights_order::iofz,
             const uint32_t direction = 0,
             const padding& output_padding = padding(),
             const int num_outputs = 1)
        : primitive_base(id, \
            filter_empty_id({x, initial_hidden_state, initial_cell_state, seq_lenghts, W, R, B, input_info(out1_prim_id), input_info(out2_prim_id)}), \
             num_outputs, {optional_data_type()}, {output_padding}),
          out1_prim_id(out1_prim_id),
          out2_prim_id(out2_prim_id),
          clip(clip),
          activations(activations),
          activation_params(activation_params),
          offset_order(offset_order),
          direction(direction) {}

    /// @brief Primitive id containing the initial value of the cell state data.
    primitive_id out1_prim_id;
    primitive_id out2_prim_id;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip;
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
        seed = hash_combine(seed, out1_prim_id);
        seed = hash_combine(seed, clip);
        seed = hash_range(seed, activations.begin(), activations.end());
        for (auto& act_param : activation_params) {
            seed = hash_combine(seed, act_param.a);
            seed = hash_combine(seed, act_param.b);
        }
        seed = hash_combine(seed, offset_order);
        seed = hash_combine(seed, direction);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rnn_seq>(rhs);

        bool act_params_eq = activation_params.size() == rhs_casted.activation_params.size();
        for (size_t i = 0; i < activation_params.size(); ++i) {
            act_params_eq &= activation_params[i].a == rhs_casted.activation_params[i].a &&
                             activation_params[i].b == rhs_casted.activation_params[i].b;
        }

        #define cmp_fields(name) name == rhs_casted.name
        return act_params_eq &&
               cmp_fields(out1_prim_id) &&
               cmp_fields(out2_prim_id) &&
               cmp_fields(clip) &&
               cmp_fields(activations) &&
               cmp_fields(offset_order) &&
               cmp_fields(direction);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rnn_seq>::save(ob);
        ob << out1_prim_id;
        ob << out2_prim_id;
        ob << clip;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << direction;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rnn_seq>::load(ib);
        ib >> out1_prim_id;
        ib >> out2_prim_id;
        ib >> clip;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> direction;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        return ret;
    }
    std::vector<input_info> filter_empty_id(std::vector<input_info> inputs_info) {
        std::vector<input_info> out;
        for (const auto& input_info : inputs_info) {
            if (!input_info.pid.empty()) {
                out.emplace_back(input_info);
            }
        }
        return out;
    }
};
} //namespace cldnn