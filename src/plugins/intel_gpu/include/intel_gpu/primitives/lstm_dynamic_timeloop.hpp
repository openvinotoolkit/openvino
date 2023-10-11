// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs forward calcaulations of input gates for dynamic lstm layer.
/// @details The current implementation of LSTM_DYNAMIC is described the following equations.
///   it = f(Xt*(Wi^T) + Ht-1*Ri + Wbi)
///   ft = f(Xt*(Wf^T) + Ht-1*Rf + Wbf)
///   ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc)
///   Ct = ft (.) Ct-1 + it (.) ct
///   ot = f(Xt*(Wo^T) + Ht-1*Ro + Wbo)
///   Ht = ot (.) h(Ct)
/// Where f = Sigmoid, g = Tanh, and h = Tanh.
struct lstm_dynamic_timeloop
    : public primitive_base<lstm_dynamic_timeloop> {
    CLDNN_DECLARE_PRIMITIVE(lstm_dynamic_timeloop)

    lstm_dynamic_timeloop() : primitive_base("", {}) {}

    /// @brief Constructs lstm_dynamic layer.
    /// @param id This primitive id.
    /// @param input Primitive id of input layer.
    /// @param dyn_length Primitive id of ilayer containg dynamic length values (shape: 1D).
    /// @param recurrent Primitive id containing recurrent data.
    /// @param initial_hidden Primitive id containing initial_hidden data. Provide empty string if using lstm_dynamic without initial_hidden values.
    /// @param initial_cell Primitive id containing initial_cell data. Provide empty string if using lstm_dynamic without initial_cell values.
    /// @param clip Clip threshold. Provide 0 if using lstm without activations clip threshold.
    /// @param input_forget Provide 0 if using lstm without coupled input-forget gates.
    lstm_dynamic_timeloop(const primitive_id& id,
                          const input_info& input,
                          const primitive_id& dyn_length,
                          const primitive_id& recurrent,
                          const primitive_id& last_hidden_state = "",
                          const primitive_id& last_cell_state = "",
                          const primitive_id& initial_hidden = "",
                          const primitive_id& initial_cell = "",
                          const float clip = 0.0f,
                          const bool input_forget = 0,
                          const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          dyn_length(dyn_length),
          recurrent(recurrent),
          last_hidden_state(last_hidden_state),
          last_cell_state(last_cell_state),
          initial_hidden(initial_hidden),
          initial_cell(initial_cell),
          clip(clip),
          input_forget(input_forget) {}

    /// @brief Primitive id containing the dynamic sequence lengths.
    primitive_id dyn_length;
    /// @brief Primitive id containing recurrent data.
    primitive_id recurrent;
    /// @brief Primitive Id of mutable data primitive pointing to buffer, which will be filled with last hidden state.
    primitive_id last_hidden_state;
    /// @brief Primitive Id of mutable data primitive pointing to buffer, which will be filled with last cell state.
    primitive_id last_cell_state;
    /// @brief Primitive id containing the initial value of the hidden data.
    primitive_id initial_hidden;
    /// @brief Array of primitive ids containing the initial value of the hidden state data (Ht-1).
    primitive_id initial_cell;
    /// @brief Cell clip threshold T. It is applied to the input of activations [-T, T]. No clip is applied if it is not specified.
    float clip = 0.0f;
    /// @brief Couple the input and forget gates if input_forget is 1. Default is 0.
    bool input_forget = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, clip);
        seed = hash_combine(seed, input_forget);
        seed = hash_combine(seed, last_hidden_state.empty());
        seed = hash_combine(seed, last_cell_state.empty());
        seed = hash_combine(seed, initial_hidden.empty());
        seed = hash_combine(seed, initial_cell.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lstm_dynamic_timeloop>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(clip) &&
               cmp_fields(input_forget) &&
               cmp_fields(last_hidden_state.empty()) &&
               cmp_fields(last_cell_state.empty()) &&
               cmp_fields(initial_hidden.empty()) &&
               cmp_fields(initial_cell.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm_dynamic_timeloop>::save(ob);
        ob << dyn_length;
        ob << recurrent;
        ob << last_hidden_state;
        ob << last_cell_state;
        ob << initial_hidden;
        ob << initial_cell;
        ob << clip;
        ob << input_forget;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm_dynamic_timeloop>::load(ib);
        ib >> dyn_length;
        ib >> recurrent;
        ib >> last_hidden_state;
        ib >> last_cell_state;
        ib >> initial_hidden;
        ib >> initial_cell;
        ib >> clip;
        ib >> input_forget;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(dyn_length);
        ret.push_back(recurrent);

        if (!last_hidden_state.empty()) {
            ret.push_back(last_hidden_state);
        }
        if (!last_cell_state.empty()) {
            ret.push_back(last_cell_state);
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
}  // namespace cldnn
