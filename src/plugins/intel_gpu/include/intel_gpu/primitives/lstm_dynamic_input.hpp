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
struct lstm_dynamic_input : public primitive_base<lstm_dynamic_input> {
    CLDNN_DECLARE_PRIMITIVE(lstm_dynamic_input)

    lstm_dynamic_input() : primitive_base("", {}) {}

    /// @brief Constructs lstm_dynamic layer.
    /// @param id This primitive id.
    /// @param input Primitive id of input layer.
    /// @param dyn_length Primitive id of ilayer containg dynamic length values (shape: 1D).
    /// @param weights Primitive id containing weights data.
    /// @param recurrent Primitive id containing recurrent data.
    /// @param bias Primitive id containing bias data. Provide empty string if using lstm_dynamic without bias.
    lstm_dynamic_input(const primitive_id& id,
                       const input_info& input,
                       const primitive_id& dyn_length,
                       const primitive_id& weights,
                       const primitive_id& bias = "",
                       const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), dyn_length(dyn_length), weights(weights), bias(bias) {}

    /// @brief Primitive id containing the dynamic sequence lengths.
    primitive_id dyn_length;
    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing bias data.
    primitive_id bias;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, bias.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const lstm_dynamic_input>(rhs);

        return bias.empty() == rhs_casted.bias.empty();
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm_dynamic_input>::save(ob);
        ob << dyn_length;
        ob << weights;
        ob << bias;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm_dynamic_input>::load(ib);
        ib >> dyn_length;
        ib >> weights;
        ib >> bias;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(dyn_length);
        ret.push_back(weights);

        if (!bias.empty()) {
            ret.push_back(bias);
        }
        return ret;
    }
};
}  // namespace cldnn
