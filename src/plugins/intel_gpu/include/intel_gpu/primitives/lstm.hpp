// Copyright (C) 2018-2023 Intel Corporation
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
/// IE order: fizo
enum class lstm_weights_order {
    iofz,
    ifoz,
    izof,
    fizo
};

/// @brief LSTM Output selection
/// @details The current implementation allows the use to select the output
/// of an LSTM node by specifing any of the following options
enum class lstm_output_selection {
    /// output the entire hidden sequence
    sequence = 0,
    /// output just the last hidden value
    hidden,
    /// output the last hidden and last cell values
    hidden_cell,
    /// output the hidden sequence concatenated with the last cell
    sequence_cell
};

/// @brief Performs forward Long Short-Term Memory (LSTM) layer.
/// @details The current implementation of LSTM is described the following equations.
///   it = f(Xt*(Wi^T) + Ht-1*Ri + Wbi)
///   ft = f(Xt*(Wf^T) + Ht-1*Rf + Wbf)
///   ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc)
///   Ct = ft (.) Ct-1 + it (.) ct
///   ot = f(Xt*(Wo^T) + Ht-1*Ro + Wbo)
///   Ht = ot (.) h(Ct)
/// Where f = Sigmoid, g = Tanh, and h = Tanh.
struct lstm : public primitive_base<lstm> {
    CLDNN_DECLARE_PRIMITIVE(lstm)

    lstm() : primitive_base("", {}) {}

    /// @brief Constructs lstm layer.
    /// @param id This primitive id.
    /// @param input Vector of primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data. Provide empty string if using lstm without bias.
    /// @param initial_hidden Primitive id containing initial_hidden data. Provide empty string if using lstm without initial_hidden values.
    /// @param initial_cell Primitive id containing initial_cell data. Provide empty string if using lstm without initial_cell values.
    /// @param peepholes Primitive id containing peepholes data. Provide empty string if using lstm without peepholes.
    /// @param clip Clip threshold. Provide 0 if using lstm without activations clip threshold.
    /// @param input_forget Provide 0 if using lstm without coupled input-forget gates.
    /// @param activations Vector of activations. Specify [f, g, h]. Default are [sigmoid, tanh, tanh]
    /// @param activation_params Vector of ativation params. Specify params for each [f, g, h] activation.
    /// @brief Output selection. Default the entire hidden sequence is returned.
    /// @param offset_order Order of the concatenated weights, recurrent, and bias. ONNX default is iofz [input, output, forget, block].
    lstm(const primitive_id& id,
         const std::vector<input_info>& input,
         const primitive_id& weights,
         const primitive_id& recurrent,
         const primitive_id& bias = "",
         const primitive_id& initial_hidden = "",
         const primitive_id& initial_cell = "",
         const primitive_id& peepholes = "",
         const float clip = 0,
         const bool input_forget = 0,
         const std::vector<activation_func>& activations = {},
         const std::vector<activation_additional_params> activation_params = {},
         const lstm_output_selection output_selection = lstm_output_selection::sequence,
         const lstm_weights_order offset_order = lstm_weights_order::iofz,
         const padding& output_padding = padding())
        : primitive_base(id, input, {output_padding}),
          weights(weights),
          recurrent(recurrent),
          bias(bias),
          initial_hidden(initial_hidden),
          initial_cell(initial_cell),
          peepholes(peepholes),
          clip(clip),
          input_forget(input_forget),
          activations(activations),
          activation_params(activation_params),
          output_selection(output_selection),
          offset_order(offset_order) {}

    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing recurrent data.
    primitive_id recurrent;
    /// @brief Primitive id containing bias data.
    primitive_id bias;
    /// @brief Primitive id containing the initial value of the hidden data.
    primitive_id initial_hidden;
    /// @brief Primitive id containing the initial value of the cell state data.
    primitive_id initial_cell;
    /// @brief Primitive id containing peepholes data.
    primitive_id peepholes;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip = 0.0f;
    /// @brief Couple the input and forget gates if input_forget is 1. Default is 0.
    bool input_forget = 0;
    /// @brief A list of 3 activation functions for the input, output, forget, cell, and hidden.
    std::vector<activation_func> activations;
    /// @brief Optional scaling values used by some activation functions. The values are consumed in the order of activation functions.
    std::vector<activation_additional_params> activation_params;
    /// @brief Output selection. Default the entire hidden sequence is returned.
    lstm_output_selection output_selection = lstm_output_selection::sequence;
    /// @brief Weights, recurrent weights, and biases order. [iofz] : ONNX, [ifoz] : Caffe
    lstm_weights_order offset_order = lstm_weights_order::izof;

    // NOT SUPPORTED YET
    // /// @brief Optional tensor specifying lengths of the sequences in a batch.
    // /// If not specified - assumed all sequences in the batch to have length `seq_length`. It has shape `[batch_size]`.
    // tensor sequence_lens;
    // /// @brief The sequence output for the hidden.
    // uint32_t output_sequence;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, peepholes.empty());
        seed = hash_combine(seed, clip);
        seed = hash_combine(seed, input_forget);
        seed = hash_range(seed, activations.begin(), activations.end());
        for (auto& act_param : activation_params) {
            seed = hash_combine(seed, act_param.a);
            seed = hash_combine(seed, act_param.b);
        }
        seed = hash_combine(seed, output_selection);
        seed = hash_combine(seed, offset_order);
        seed = hash_combine(seed, bias.empty());
        seed = hash_combine(seed, initial_hidden.empty());
        seed = hash_combine(seed, initial_cell.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lstm>(rhs);

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
               cmp_fields(output_selection) &&
               cmp_fields(offset_order) &&
               cmp_fields(initial_hidden.empty()) &&
               cmp_fields(initial_cell.empty()) &&
               cmp_fields(peepholes.empty()) &&
               cmp_fields(bias.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm>::save(ob);
        ob << weights;
        ob << recurrent;
        ob << bias;
        ob << initial_hidden;
        ob << initial_cell;
        ob << peepholes;
        ob << clip;
        ob << input_forget;
        ob << activations;
        ob << activation_params;
        ob << make_data(&output_selection, sizeof(lstm_output_selection));
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm>::load(ib);
        ib >> weights;
        ib >> recurrent;
        ib >> bias;
        ib >> initial_hidden;
        ib >> initial_cell;
        ib >> peepholes;
        ib >> clip;
        ib >> input_forget;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&output_selection, sizeof(lstm_output_selection));
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(weights);
        ret.push_back(recurrent);
        if (!bias.empty()) {
            ret.push_back(bias);
        }
        if (!initial_hidden.empty()) {
            ret.push_back(initial_hidden);
        }
        if (!initial_cell.empty()) {
            ret.push_back(initial_cell);
        }
        return ret;
    }
};

struct lstm_gemm : public primitive_base<lstm_gemm> {
    CLDNN_DECLARE_PRIMITIVE(lstm_gemm)

    lstm_gemm() : primitive_base("", {}),
                  direction(0) {}

    /// @brief Constructs lstm layer.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input weights Primitive id containing weights data.
    /// @param input recurrent Primitive id containing recurrent data. It is required even for no hidden values.
    /// @param input bias Primitive id containing bias data. Provide empty string if using lstm without bias.
    /// @param input hidden Primitive id containing hidden data. Provide empty string if using lstm without hidden values.
    /// @param direction default = 0, bidirectional = 1.
    lstm_gemm(const primitive_id& id,
              const input_info& input,
              const primitive_id& weights,
              const primitive_id& recurrent,
              const primitive_id& bias = "",
              const primitive_id& hidden = "",
              const uint32_t direction = 0,
              const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          weights(weights),
          recurrent(recurrent),
          bias(bias),
          hidden(hidden),
          direction(direction) {}

    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing recurrent data.
    primitive_id recurrent;
    /// @brief Primitive id containing bias data.
    primitive_id bias;
    /// @brief Primitive id containing the initial value of the hidden data.
    primitive_id hidden;
    /// @brief direction default = 0, bidirectional = 1.
    uint32_t direction;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, direction);
        seed = hash_combine(seed, bias.empty());
        seed = hash_combine(seed, hidden.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lstm_gemm>(rhs);

        return direction == rhs_casted.direction &&
               bias.empty() == rhs_casted.bias.empty() &&
               hidden.empty() == rhs_casted.hidden.empty();
    }

    void save(BinaryOutputBuffer& ob) const override {
        ob << weights;
        ob << recurrent;
        ob << bias;
        ob << hidden;
        ob << direction;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> weights;
        ib >> recurrent;
        ib >> bias;
        ib >> hidden;
        ib >> direction;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(weights);
        ret.push_back(recurrent);
        if (!bias.empty())
            ret.push_back(bias);
        if (!hidden.empty())
            ret.push_back(hidden);
        return ret;
    }
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
             const uint32_t direction = 0,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
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
        ob << cell;
        ob << clip;
        ob << input_forget;
        ob << activations;
        ob << activation_params;
        ob << make_data(&offset_order, sizeof(lstm_weights_order));
        ob << direction;
    }

    void load(BinaryInputBuffer& ib) override {
        ib >> cell;
        ib >> clip;
        ib >> input_forget;
        ib >> activations;
        ib >> activation_params;
        ib >> make_data(&offset_order, sizeof(lstm_weights_order));
        ib >> direction;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        if (!cell.empty())
            ret.push_back(cell);
        return ret;
    }
};

}  // namespace cldnn
