// Copyright (C) 2018-2023 Intel Corporation
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

    /// @brief Constructs fully connected layer.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    /// @param bias Primitive id containing bias data. Provide empty string if using Relu without bias.
    fully_connected(const primitive_id& id,
                    const input_info& input,
                    const primitive_id& weights,
                    const primitive_id& bias = "",
                    const padding& output_padding = padding(),
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, {input}, {output_padding}),
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
                    const padding& output_padding = padding(),
                    const size_t input_size = 2,
                    const size_t weights_rank = 2)
        : primitive_base(id, { input }, {output_padding}, {optional_data_type{data_type}}),
          weights(weights),
          bias(bias),
          input_size(input_size),
          weights_rank(weights_rank)
    {}

    /// @brief Primitive id containing weights data.
    primitive_id weights;
    /// @brief Primitive id containing bias data.
    primitive_id bias;
    /// @brief Primitive dimension size.
    size_t input_size;
    /// @brief Primitive weights rank.
    size_t weights_rank;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, input_size);
        seed = hash_combine(seed, weights_rank);
        seed = hash_combine(seed, bias.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const fully_connected>(rhs);

        return input_size == rhs_casted.input_size &&
               weights_rank == rhs_casted.weights_rank &&
               bias.empty() == rhs_casted.bias.empty();
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(weights);

        if (!bias.empty())
            ret.push_back(bias);

        return ret;
    }
};
}  // namespace cldnn
