// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs forward fully connected layer (inner product).
/// Also supports built-in Relu @CLDNN_PRIMITIVE_DESC{activation} available by setting it in arguments.
/// @notes
/// - Equation: Input[F x Y x F] x Output(X) == Weights(B x F x X x F) has to be fulfilled
/// - Bias has to be linear data [1,1,1,X], where X is equal to number of outputs.

/// <table>
/// <caption id = "multi_row">Format support</caption>
///        <tr><th>Data type               <th>activation format       <th>weights format
///        <tr><td rowspan="7">F32         <td rowspan="4">bfyx        <td>yxfb
///        <tr>                                                        <td>fyxb
///        <tr>                                                        <td>bs_fs_fsv8_bsv8
///        <tr>                                                        <td>bs_f_bsv16
///        <tr>                            <td rowspan="3">yxfb        <td>bfyx
///        <tr>                                                        <td>yxfb
///        <tr>                                                        <td>bs_fs_fsv8_bsv8
///        <tr><td rowspan="4">F16         <td rowspan="3">bfyx        <td>yxfb
///        <tr>                                                        <td>fyxb
///        <tr>                                                        <td>bs_f_bsv16
///        <tr>                            <td >yxfb                   <td>bfyx
/// </table>

struct fully_connected : public primitive_base<fully_connected> {
    CLDNN_DECLARE_PRIMITIVE(fully_connected)

    fully_connected() : primitive_base("", {}) {}

    /// @brief Constructs fully connected layer.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data. Provide empty string if using Relu without bias.
    fully_connected(const primitive_id& id,
                    const input_info& input,
                    const primitive_id& weights,
                    const primitive_id& bias = "",
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, {input}),
          weights(weights),
          bias(bias),
          input_size(input_size),
          weights_rank(weights_rank)
    {}

    /// @brief Constructs fully connected layer.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data. Provide empty string if using Relu without bias.
    fully_connected(const primitive_id& id,
                    const input_info& input,
                    const primitive_id& weights,
                    const primitive_id& bias,
                    const data_types data_type,
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, { input }, 1, {optional_data_type{data_type}}),
          weights(weights),
          bias(bias),
          input_size(input_size),
          weights_rank(weights_rank)
    {}

    /// @brief Constructs fully connected compressed layer.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data.
    /// @param compression_scale Primitive id containing scale factors for weights decompression.
    /// @param compression_zero_point Primitive id containing zero points for weights decompression.
    fully_connected(const primitive_id& id,
                    const input_info& input,
                    const primitive_id& weights,
                    const primitive_id& bias,
                    const primitive_id& decompression_scale,
                    const primitive_id& decompression_zero_point,
                    const data_types data_type,
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, { input }, 1, {optional_data_type{data_type}}),
          weights(weights),
          bias(bias),
          compressed_weights(true),
          decompression_scale(decompression_scale),
          decompression_zero_point(decompression_zero_point),
          dynamic_quantized_activation(false),
          dynamic_quantized_activation_zp(false),
          input_size(input_size),
          weights_rank(weights_rank) {
        OPENVINO_ASSERT(!decompression_scale.empty(), "[GPU] Compressed fully connected requires at least decompression scale input");
    }

    /// @brief Constructs fully connected compressed layer.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data.
    /// @param compression_scale Primitive id containing scale factors for weights decompression.
    /// @param compression_zero_point Primitive id containing zero points for weights decompression.
    /// @param activation_scale Primitive id containing scale factor for activation.
    /// @param activation_zero_point Primitive id containing zero point for activation.
    fully_connected(const primitive_id& id,
                    const input_info& input,
                    const primitive_id& weights,
                    const primitive_id& bias,
                    const primitive_id& decompression_scale,
                    const primitive_id& decompression_zero_point,
                    const input_info& activation_scale,
                    const input_info& activation_zero_point,
                    const data_types data_type,
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, { input }, 1, {optional_data_type{data_type}}),
          weights(weights),
          bias(bias),
          compressed_weights(true),
          decompression_scale(decompression_scale),
          decompression_zero_point(decompression_zero_point),
          dynamic_quantized_activation(false),
          dynamic_quantized_activation_zp(false),
          activation_scale(activation_scale),
          activation_zero_point(activation_zero_point),
          input_size(input_size),
          weights_rank(weights_rank) {
        if (activation_scale.is_valid())
            dynamic_quantized_activation = true;
        if (activation_zero_point.is_valid())
            dynamic_quantized_activation_zp = true;

        OPENVINO_ASSERT(!decompression_scale.empty(), "[GPU] Compressed fully connected requires at least decompression scale input");
    }

    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing bias data.
    primitive_id bias;

    bool compressed_weights = false;
    primitive_id decompression_scale = "";
    primitive_id decompression_zero_point = "";
    bool dynamic_quantized_activation = false;
    bool dynamic_quantized_activation_zp = false;
    input_info activation_scale = {"", 0};
    input_info activation_zero_point = {"", 0};
    std::optional<float> decompression_zero_point_scalar = std::optional<float>();

    /// @brief Primitive dimension size.
    size_t input_size = 2;
    /// @brief Primitive weights rank.
    size_t weights_rank = 2;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, input_size);
        seed = hash_combine(seed, weights_rank);
        seed = hash_combine(seed, bias.empty());
        seed = hash_combine(seed, compressed_weights);
        seed = hash_combine(seed, !decompression_scale.empty());
        seed = hash_combine(seed, !decompression_zero_point.empty());
        seed = hash_combine(seed, activation_scale.is_valid());
        seed = hash_combine(seed, activation_zero_point.is_valid());
        seed = hash_combine(seed, decompression_zero_point_scalar.has_value());
        seed = hash_combine(seed, decompression_zero_point_scalar.value_or(0.0f));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const fully_connected>(rhs);

        return input_size == rhs_casted.input_size &&
               weights_rank == rhs_casted.weights_rank &&
               bias.empty() == rhs_casted.bias.empty() &&
               compressed_weights == rhs_casted.compressed_weights &&
               decompression_scale.empty() == rhs_casted.decompression_scale.empty() &&
               decompression_zero_point.empty() == rhs_casted.decompression_zero_point.empty() &&
               activation_scale.is_valid() == rhs_casted.activation_scale.is_valid() &&
               activation_zero_point.is_valid() == rhs_casted.activation_zero_point.is_valid() &&
               decompression_zero_point_scalar.value_or(0.0f) == rhs_casted.decompression_zero_point_scalar.value_or(0.0f);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fully_connected>::save(ob);
        ob << weights;
        ob << bias;
        ob << compressed_weights;
        ob << decompression_scale;
        ob << decompression_zero_point;
        ob << activation_scale;
        ob << activation_zero_point;
        ob << input_size;
        ob << weights_rank;
        ob << dynamic_quantized_activation;
        ob << dynamic_quantized_activation_zp;

        if (decompression_zero_point_scalar.has_value()) {
            ob << true;
            float decompression_zero_point_value = decompression_zero_point_scalar.value();
            ob << decompression_zero_point_value;
        } else {
            ob << false;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fully_connected>::load(ib);
        ib >> weights;
        ib >> bias;
        ib >> compressed_weights;
        ib >> decompression_scale;
        ib >> decompression_zero_point;
        ib >> activation_scale;
        ib >> activation_zero_point;
        ib >> input_size;
        ib >> weights_rank;
        ib >> dynamic_quantized_activation;
        ib >> dynamic_quantized_activation_zp;

        bool has_value;
        ib >> has_value;
        if (has_value) {
            float decompression_zero_point_value = 0.f;
            ib >> decompression_zero_point_value;
            decompression_zero_point_scalar = decompression_zero_point_value;
        } else {
            decompression_zero_point_scalar = std::optional<float>();
        }
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        ret.push_back(weights);

        if (!bias.empty())
            ret.push_back(bias);

        if (!decompression_scale.empty())
            ret.push_back(decompression_scale);

        if (!decompression_zero_point.empty())
            ret.push_back(decompression_zero_point);

        if (activation_scale.is_valid())
            ret.push_back(activation_scale);

        if (activation_zero_point.is_valid())
            ret.push_back(activation_zero_point);

        return ret;
    }
};
}  // namespace cldnn
