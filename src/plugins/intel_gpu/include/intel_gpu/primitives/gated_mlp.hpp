// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/glu.hpp"
#include "primitive.hpp"

namespace cldnn {

struct gated_mlp : public primitive_base<gated_mlp> {
    CLDNN_DECLARE_PRIMITIVE(gated_mlp)

    gated_mlp() : primitive_base("", {}) {}

    gated_mlp(const primitive_id& id,
              const input_info& src,
              const input_info& w_gate,
              const input_info& w_up,
              const input_info& w_down,
              ov::op::internal::GLU::GluType activation,
              const tensor& output_size,
              const data_types output_dt)
        : primitive_base(id, {src}, 1, {optional_data_type{output_dt}}),
          weights_gate(w_gate),
          weights_up(w_up),
          weights_down(w_down),
          activation(activation),
          output_size(output_size) {}

    input_info weights_gate;
    input_info weights_up;
    input_info weights_down;
    ov::op::internal::GLU::GluType activation = ov::op::internal::GLU::GluType::Swish;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, static_cast<size_t>(activation));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const gated_mlp>(rhs);
        return activation == rhs_casted.activation;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gated_mlp>::save(ob);
        ob << weights_gate;
        ob << weights_up;
        ob << weights_down;
        ob << make_data(&activation, sizeof(activation));
        ob << output_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_mlp>::load(ib);
        ib >> weights_gate;
        ib >> weights_up;
        ib >> weights_down;
        ib >> make_data(&activation, sizeof(activation));
        ib >> output_size;
    }

protected:
    std::map<size_t, const input_info*> get_dependencies_map() const override {
        auto ret = std::map<size_t, const input_info*>{};
        auto idx = input.size();
        ret[idx++] = &weights_gate;
        ret[idx++] = &weights_up;
        ret[idx++] = &weights_down;
        return ret;
    }
};

}  // namespace cldnn
