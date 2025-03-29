// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include <string>
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

template <typename PType>
struct RNNParams : public primitive_base<PType> {
    RNNParams() : primitive_base<PType>("", {}) {}
    RNNParams(const RNNParams&) = default;
    RNNParams(const primitive_id& id,
              const input_info& x,
              const input_info& initial_hidden_state,
              const input_info& initial_cell_state,
              const input_info& W,
              const input_info& R,
              const input_info& B,
              const input_info& seq_lenghts,
              const float clip = 0,
              bool input_forget = false,
              const std::vector<activation_func>& activations = {activation_func::logistic,
                                                                activation_func::hyperbolic_tan,
                                                                activation_func::hyperbolic_tan},
              const std::vector<activation_additional_params>& activation_params = {},
              const lstm_weights_order& offset_order = lstm_weights_order::iofz,
              const ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
              const int num_outputs = 1)
        : primitive_base<PType>(id, {x}, num_outputs),
        x(x),
        initial_hidden_state(initial_hidden_state),
        initial_cell_state(initial_cell_state),
        W(W),
        R(R),
        B(B),
        seq_lenghts(seq_lenghts),
        clip(clip),
        input_forget(input_forget),
        activations(activations),
        activation_params(activation_params),
        offset_order(offset_order),
        direction(direction) {
        std::vector<std::string> pids{initial_hidden_state.pid, initial_cell_state.pid, W.pid, R.pid, B.pid, seq_lenghts.pid};
        for (auto pid : pids) {
            if (!pid.empty()) {
                primitive_base<PType>::input.push_back(pid);
            }
        }
    }

    input_info x;
    input_info initial_hidden_state;
    input_info initial_cell_state;
    input_info W;
    input_info R;
    input_info B;
    input_info seq_lenghts;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip;
    bool input_forget;
    /// @brief A list of 3 activation functions for the input, output, forget, cell, and hidden.
    std::vector<activation_func> activations;
    /// @brief Optional scaling values used by some activation functions. The values are consumed in the order of activation functions.
    std::vector<activation_additional_params> activation_params;
    /// @brief Weights, recurrent weights, and biases order. [iofz] : ONNX, [ifoz] : Caffe
    lstm_weights_order offset_order;
    /// @brief direction of LSTMSequence - only FORWARD or REVERSE, currently BIDIRECTIONAL not supported
    ov::op::RecurrentSequenceDirection direction;

    int num_directions() const {
        return direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, !x.pid.empty());
        seed = hash_combine(seed, !initial_hidden_state.pid.empty());
        seed = hash_combine(seed, !initial_cell_state.pid.empty());
        seed = hash_combine(seed, !seq_lenghts.pid.empty());
        seed = hash_combine(seed, !W.pid.empty());
        seed = hash_combine(seed, !R.pid.empty());
        seed = hash_combine(seed, !B.pid.empty());
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
        if (!primitive::compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const PType>(rhs);
        bool act_params_eq = activation_params.size() == rhs_casted.activation_params.size();
        for (size_t i = 0; i < activation_params.size(); ++i) {
            act_params_eq &= activation_params[i].a == rhs_casted.activation_params[i].a &&
                             activation_params[i].b == rhs_casted.activation_params[i].b;
        }

        #define cmp_fields(name) name == rhs_casted.name
        return act_params_eq &&
               cmp_fields(x.pid.empty()) &&
               cmp_fields(initial_hidden_state.pid.empty()) &&
               cmp_fields(initial_cell_state.pid.empty()) &&
               cmp_fields(seq_lenghts.pid.empty()) &&
               cmp_fields(W.pid.empty()) &&
               cmp_fields(R.pid.empty()) &&
               cmp_fields(B.pid.empty()) &&
               cmp_fields(clip) &&
               cmp_fields(activations) &&
               cmp_fields(offset_order) &&
               cmp_fields(direction);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<PType>::save(ob);
        ob << x;
        ob << initial_hidden_state;
        ob << initial_cell_state;
        ob << W;
        ob << R;
        ob << B;
        ob << seq_lenghts;
        ob << clip;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << make_data(&direction, sizeof(ov::op::RecurrentSequenceDirection));
    }

    void load(BinaryInputBuffer& ib) override{
        primitive_base<PType>::load(ib);
        ib >> x;
        ib >> initial_hidden_state;
        ib >> initial_cell_state;
        ib >> W;
        ib >> R;
        ib >> B;
        ib >> seq_lenghts;
        ib >> clip;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> make_data(&direction, sizeof(ov::op::RecurrentSequenceDirection));
    }
};

struct lstm_seq : public RNNParams<lstm_seq> {
    CLDNN_DECLARE_PRIMITIVE(lstm_seq)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    lstm_seq() : RNNParams() {
        weights = W.pid;
        input = x.pid;
    }
    lstm_seq(const lstm_seq&) = default;
    primitive_id input;
    primitive_id weights;
};
} //namespace cldnn
