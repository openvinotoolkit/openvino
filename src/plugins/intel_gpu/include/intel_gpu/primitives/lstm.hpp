// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"
#include "rnn.hpp"


namespace cldnn {

struct lstm_elt : public primitive_base<lstm_elt> {
    CLDNN_DECLARE_PRIMITIVE(lstm_elt)

    lstm_elt() : primitive_base("", {}), input_forget(0) {
        params.clip = 0;
        params.offset_order = lstm_weights_order::iofz;
        params.direction = 0;
    }

    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;

    /// @brief Constructs lstm layer.
    /// @param RNNParam common params for rnns
    /// @param input_forget Provide 0 if using lstm without coupled input-forget gates.
    lstm_elt(const RNNParams& p, bool input_forget): primitive_base(p.id, p.get_inputs(), p.num_outputs, \
    {optional_data_type()}, {p.output_padding}), params(p), input_forget(input_forget) {}

    RNNParams params;
    bool input_forget;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, params.hash());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        auto rhs_casted = downcast<const lstm_elt>(rhs);
        return params == rhs_casted.params;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<lstm_elt>::save(ob);
        params.save(ob);
        ob << input_forget;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<lstm_elt>::load(ib);
        params.load(ib);
        ib >> input_forget;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!params.initial_cell_state.pid.empty())
            ret.push_back(params.initial_cell_state);
        return ret;
    }
};


}  // namespace cldnn
