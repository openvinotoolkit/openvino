// Copyright (C) 2018-2024 Intel Corporation
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

struct RNNParams{
    RNNParams() = default;
    RNNParams(const RNNParams&) = default;
    RNNParams(const primitive_id& id,
             const input_info& xWB,
             const input_info& initial_hidden_state,
             const input_info& initial_cell_state,
             const input_info& R,
             const input_info& seq_lenghts,
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
             const int num_outputs = 1) : id(id),
             xWB(xWB), initial_hidden_state(initial_hidden_state), initial_cell_state(initial_cell_state), R(R), seq_lenghts(seq_lenghts), \
             out1_prim_id(out1_prim_id), out2_prim_id(out2_prim_id), clip(clip), activations(activations), activation_params(activation_params), \
             offset_order(offset_order), direction(direction), output_padding(output_padding), num_outputs(num_outputs) {}
    primitive_id id;
    input_info xWB;
    input_info initial_hidden_state;
    input_info initial_cell_state;/// @brief for lstm_elt primitive field for cell input_info
    input_info R;
    input_info seq_lenghts;
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
    /// @brief direction forward = 0, reverse = 1.
    uint32_t direction;
    padding output_padding;
    int num_outputs;

    size_t hash() const {
        size_t seed = hash_combine(3, id);
        seed = hash_combine(seed, xWB.pid);
        seed = hash_combine(seed, initial_hidden_state.pid);
        seed = hash_combine(seed, initial_cell_state.pid);
        seed = hash_combine(seed, seq_lenghts.pid);
        seed = hash_combine(seed, R.pid);
        seed = hash_combine(seed, out1_prim_id);
        seed = hash_combine(seed, out2_prim_id);
        seed = hash_combine(seed, clip);
        seed = hash_range(seed, activations.begin(), activations.end());
        for (auto& act_param : activation_params) {
            seed = hash_combine(seed, act_param.a);
            seed = hash_combine(seed, act_param.b);
        }
        seed = hash_combine(seed, offset_order);
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, num_outputs);
        return seed;
    }

    bool operator==(const RNNParams& rhs) const {
        bool act_params_eq = activation_params.size() == rhs.activation_params.size();
        for (size_t i = 0; i < activation_params.size(); ++i) {
            act_params_eq &= activation_params[i].a == rhs.activation_params[i].a &&
                             activation_params[i].b == rhs.activation_params[i].b;
        }

        #define cmp_fields(name) name == rhs.name
        return act_params_eq &&
               cmp_fields(id) &&
               cmp_fields(xWB) &&
               cmp_fields(initial_hidden_state) &&
               cmp_fields(initial_cell_state) &&
               cmp_fields(seq_lenghts) &&
               cmp_fields(R) &&
               cmp_fields(out1_prim_id) &&
               cmp_fields(out2_prim_id) &&
               cmp_fields(clip) &&
               cmp_fields(activations) &&
               cmp_fields(offset_order) &&
               cmp_fields(direction) &&
               cmp_fields(output_padding) &&
               cmp_fields(num_outputs);
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const {
        ob << id;
        ob << xWB;
        ob << initial_hidden_state;
        ob << initial_cell_state;
        ob << R;
        ob << seq_lenghts;
        ob << out1_prim_id;
        ob << out2_prim_id;
        ob << clip;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << direction;
        ob << output_padding;
        ob << num_outputs;
    }

    void load(BinaryInputBuffer& ib) {
        ib >> id;
        ib >> xWB;
        ib >> initial_hidden_state;
        ib >> initial_cell_state;
        ib >> R;
        ib >> seq_lenghts;
        ib >> out1_prim_id;
        ib >> out2_prim_id;
        ib >> clip;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> direction;
        ib >> output_padding;
        ib >> num_outputs;
    }

    std::vector<input_info> get_inputs() const {
        return filter_empty_id( {xWB, initial_hidden_state, initial_cell_state, R, seq_lenghts, input_info(out1_prim_id), input_info(out2_prim_id)});
    }

protected:
    std::vector<input_info> filter_empty_id(std::vector<input_info> inputs_info) const {
        std::vector<input_info> out;
        for (const auto& input_info : inputs_info) {
            if (!input_info.pid.empty()) {
                out.emplace_back(input_info);
            }
        }
        return out;
    }
};

struct lstm_seq : public primitive_base<lstm_seq> {
    CLDNN_DECLARE_PRIMITIVE(lstm_seq)
    lstm_seq() : primitive_base("", {}) {}

    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    lstm_seq(const RNNParams& p): primitive_base(p.id, p.get_inputs(), p.num_outputs, \
    {optional_data_type()}, {p.output_padding}), params(p) {}
    RNNParams params;
    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, params.hash());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const lstm_seq>(rhs);
        return params == rhs_casted.params && output_data_types == rhs_casted.output_data_types;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm_seq>::save(ob);
        params.save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm_seq>::load(ib);
        params.load(ib);
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        return {};
    }
};

} //namespace cldnn
