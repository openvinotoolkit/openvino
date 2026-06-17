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

    /// @brief Constructs gated MLP layer with uncompressed weights.
    /// @param id This primitive id.
    /// @param src Input primitive id (activation).
    /// @param w_gate Primitive id containing gate projection weights.
    /// @param w_up Primitive id containing up projection weights.
    /// @param w_down Primitive id containing down projection weights.
    /// @param activation Activation function type (Swish, Gelu, etc.).
    /// @param output_size Expected output tensor size.
    /// @param output_dt Output data type.
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

    /// @brief Constructs gated MLP layer with compressed weights (scales and optional zero points).
    /// @param id This primitive id.
    /// @param src Input primitive id (activation).
    /// @param w_gate Primitive id containing gate projection weights.
    /// @param w_up Primitive id containing up projection weights.
    /// @param w_down Primitive id containing down projection weights.
    /// @param scale_gate Primitive id containing decompression scale for gate weights.
    /// @param scale_up Primitive id containing decompression scale for up weights.
    /// @param scale_down Primitive id containing decompression scale for down weights.
    /// @param zp_gate Primitive id containing decompression zero point for gate weights.
    /// @param zp_up Primitive id containing decompression zero point for up weights.
    /// @param zp_down Primitive id containing decompression zero point for down weights.
    /// @param activation Activation function type (Swish, Gelu, etc.).
    /// @param output_size Expected output tensor size.
    /// @param output_dt Output data type.
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
          activation(activation),
          output_size(output_size) {
        OPENVINO_ASSERT(decompression_scale_gate.is_valid() && decompression_scale_up.is_valid() && decompression_scale_down.is_valid(),
                        "GatedMLP compressed mode requires decompression scales.");
    }

    /// @brief Constructs gated MLP layer with compressed weights and dynamic quantized activation.
    /// @param id This primitive id.
    /// @param src Input primitive id (dynamically quantized activation, i8/u8).
    /// @param w_gate Primitive id containing gate projection weights.
    /// @param w_up Primitive id containing up projection weights.
    /// @param w_down Primitive id containing down projection weights.
    /// @param scale_gate Primitive id containing decompression scale for gate weights.
    /// @param scale_up Primitive id containing decompression scale for up weights.
    /// @param scale_down Primitive id containing decompression scale for down weights.
    /// @param zp_gate Primitive id containing decompression zero point for gate weights.
    /// @param zp_up Primitive id containing decompression zero point for up weights.
    /// @param zp_down Primitive id containing decompression zero point for down weights.
    /// @param act_scale Primitive id containing scale factor for activation.
    /// @param act_zp Primitive id containing zero point for activation.
    /// @param act_precomputed_reduction Primitive id containing precomputed reduction for activation.
    /// @param activation Activation function type (Swish, Gelu, etc.).
    /// @param output_size Expected output tensor size.
    /// @param output_dt Output data type.
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
              const input_info& act_scale,
              const input_info& act_zp,
              const input_info& act_precomputed_reduction,
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
          activation_scale(act_scale),
          activation_zero_point(act_zp),
          activation_precomputed_reduction(act_precomputed_reduction),
          compressed_weights(true),
          activation(activation),
          output_size(output_size) {
        OPENVINO_ASSERT(decompression_scale_gate.is_valid() && decompression_scale_up.is_valid() && decompression_scale_down.is_valid(),
                        "GatedMLP compressed mode requires decompression scales.");
        if (activation_scale.is_valid())
            dynamic_quantized_activation = true;
        if (activation_zero_point.is_valid())
            dynamic_quantized_activation_zp = true;
        if (activation_precomputed_reduction.is_valid())
            dynamic_quantized_precomputed_reduction = true;
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
    input_info activation_scale;
    input_info activation_zero_point;
    input_info activation_precomputed_reduction;
    bool compressed_weights = false;
    bool dynamic_quantized_activation = false;
    bool dynamic_quantized_activation_zp = false;
    bool dynamic_quantized_precomputed_reduction = false;
    ov::op::internal::GLU::GluType activation = ov::op::internal::GLU::GluType::Swish;
    tensor output_size;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, compressed_weights);
        seed = hash_combine(seed, !decompression_scale_gate.is_valid());
        seed = hash_combine(seed, !decompression_zero_point_gate.is_valid());
        seed = hash_combine(seed, activation_scale.is_valid());
        seed = hash_combine(seed, activation_zero_point.is_valid());
        seed = hash_combine(seed, activation_precomputed_reduction.is_valid());
        seed = hash_combine(seed, static_cast<size_t>(activation));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const gated_mlp>(rhs);
        return activation == rhs_casted.activation &&
               compressed_weights == rhs_casted.compressed_weights &&
               decompression_scale_gate.is_valid() == rhs_casted.decompression_scale_gate.is_valid() &&
               decompression_zero_point_gate.is_valid() == rhs_casted.decompression_zero_point_gate.is_valid() &&
               activation_scale.is_valid() == rhs_casted.activation_scale.is_valid() &&
               activation_zero_point.is_valid() == rhs_casted.activation_zero_point.is_valid() &&
               activation_precomputed_reduction.is_valid() == rhs_casted.activation_precomputed_reduction.is_valid();
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
        ob << activation_scale;
        ob << activation_zero_point;
        ob << activation_precomputed_reduction;
        ob << compressed_weights;
        ob << dynamic_quantized_activation;
        ob << dynamic_quantized_activation_zp;
        ob << dynamic_quantized_precomputed_reduction;
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
        ib >> activation_scale;
        ib >> activation_zero_point;
        ib >> activation_precomputed_reduction;
        ib >> compressed_weights;
        ib >> dynamic_quantized_activation;
        ib >> dynamic_quantized_activation_zp;
        ib >> dynamic_quantized_precomputed_reduction;
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

        if (activation_scale.is_valid())
            ret[idx++] = &activation_scale;
        if (activation_zero_point.is_valid())
            ret[idx++] = &activation_zero_point;
        if (activation_precomputed_reduction.is_valid())
            ret[idx++] = &activation_precomputed_reduction;

        return ret;
    }
};

}  // namespace cldnn
