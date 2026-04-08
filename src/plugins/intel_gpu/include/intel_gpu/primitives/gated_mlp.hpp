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

    gated_mlp(const primitive_id& id,
              const input_info& src,
              const input_info& w_gate,
              const input_info& w_up,
              const input_info& w_down,
              const input_info& scale_gate,
              const input_info& scale_up,
              const input_info& scale_down,
              ov::op::internal::GLU::GluType activation,
              const tensor& output_size,
              const data_types output_dt)
        : primitive_base(id, {src}, 1, {optional_data_type{output_dt}}),
          weights_gate(w_gate),
          weights_up(w_up),
          weights_down(w_down),
          decompression_scale_gate(scale_gate),
          decompression_scale_up(scale_up),
          decompression_scale_down(scale_down),
          compressed_weights(true),
          activation(activation),
          output_size(output_size) {
        OPENVINO_ASSERT(decompression_scale_gate.is_valid() && decompression_scale_up.is_valid() && decompression_scale_down.is_valid(),
                        "GatedMLP compressed mode requires decompression scales.");
    }

    gated_mlp(const primitive_id& id,
              const input_info& src,
              const input_info& w_gate,
              const input_info& w_up,
              const input_info& w_down,
              const input_info& scale_gate,
              const input_info& scale_up,
              const input_info& scale_down,
              const input_info& zp_gate,
              const input_info& zp_up,
              const input_info& zp_down,
              ov::op::internal::GLU::GluType activation,
              const tensor& output_size,
              const data_types output_dt)
        : primitive_base(id, {src}, 1, {optional_data_type{output_dt}}),
          weights_gate(w_gate),
          weights_up(w_up),
          weights_down(w_down),
          decompression_scale_gate(scale_gate),
          decompression_scale_up(scale_up),
          decompression_scale_down(scale_down),
          decompression_zero_point_gate(zp_gate),
          decompression_zero_point_up(zp_up),
          decompression_zero_point_down(zp_down),
          compressed_weights(true),
          has_decompression_zero_points(true),
          activation(activation),
          output_size(output_size) {
        OPENVINO_ASSERT(decompression_scale_gate.is_valid() && decompression_scale_up.is_valid() && decompression_scale_down.is_valid(),
                        "GatedMLP compressed mode requires decompression scales.");
        OPENVINO_ASSERT(decompression_zero_point_gate.is_valid() && decompression_zero_point_up.is_valid() && decompression_zero_point_down.is_valid(),
                        "GatedMLP compressed mode with zero points requires decompression zero points.");
    }

    input_info weights_gate;
    input_info weights_up;
    input_info weights_down;
    input_info decompression_scale_gate;
    input_info decompression_scale_up;
    input_info decompression_scale_down;
    input_info decompression_zero_point_gate;
    input_info decompression_zero_point_up;
    input_info decompression_zero_point_down;
    bool compressed_weights = false;
    bool has_decompression_zero_points = false;
    ov::op::internal::GLU::GluType activation = ov::op::internal::GLU::GluType::Swish;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, compressed_weights);
        seed = hash_combine(seed, has_decompression_zero_points);
        seed = hash_combine(seed, static_cast<size_t>(activation));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const gated_mlp>(rhs);
        return activation == rhs_casted.activation &&
               compressed_weights == rhs_casted.compressed_weights &&
               has_decompression_zero_points == rhs_casted.has_decompression_zero_points;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gated_mlp>::save(ob);
        ob << weights_gate;
        ob << weights_up;
        ob << weights_down;
        ob << decompression_scale_gate;
        ob << decompression_scale_up;
        ob << decompression_scale_down;
        ob << decompression_zero_point_gate;
        ob << decompression_zero_point_up;
        ob << decompression_zero_point_down;
        ob << compressed_weights;
        ob << has_decompression_zero_points;
        ob << make_data(&activation, sizeof(activation));
        ob << output_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gated_mlp>::load(ib);
        ib >> weights_gate;
        ib >> weights_up;
        ib >> weights_down;
        ib >> decompression_scale_gate;
        ib >> decompression_scale_up;
        ib >> decompression_scale_down;
        ib >> decompression_zero_point_gate;
        ib >> decompression_zero_point_up;
        ib >> decompression_zero_point_down;
        ib >> compressed_weights;
        ib >> has_decompression_zero_points;
        ib >> make_data(&activation, sizeof(activation));
        ib >> output_size;
    }

protected:
    std::map<size_t, const input_info*> get_dependencies_map() const override {
        auto ret = std::map<size_t, const input_info*>{};
        auto idx = input.size();

        OPENVINO_ASSERT(weights_gate.is_valid());
        OPENVINO_ASSERT(weights_up.is_valid());
        OPENVINO_ASSERT(weights_down.is_valid());
        ret[idx++] = &weights_gate;
        ret[idx++] = &weights_up;
        ret[idx++] = &weights_down;

        if (decompression_scale_gate.is_valid())
            ret[idx++] = &decompression_scale_gate;
        if (decompression_scale_up.is_valid())
            ret[idx++] = &decompression_scale_up;
        if (decompression_scale_down.is_valid())
            ret[idx++] = &decompression_scale_down;

        if (decompression_zero_point_gate.is_valid())
            ret[idx++] = &decompression_zero_point_gate;
        if (decompression_zero_point_up.is_valid())
            ret[idx++] = &decompression_zero_point_up;
        if (decompression_zero_point_down.is_valid())
            ret[idx++] = &decompression_zero_point_down;

        return ret;
    }
};

}  // namespace cldnn
