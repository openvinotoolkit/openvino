// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief
/// @details
struct reverse_sequence : public primitive_base<reverse_sequence> {
    CLDNN_DECLARE_PRIMITIVE(reverse_sequence)

    /// @brief Constructs reverse_sequence primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param seq_lengths Sequence lengths primitive id.
    /// @param seq_axis The axis which is partially reversed.
    /// @param batch_axis The axis along which reversal is performed.
    reverse_sequence(const primitive_id& id,
                     const input_info& input,
                     const input_info& seq_lengths,
                     const int32_t seq_axis,
                     const int32_t batch_axis = 0,
                     const primitive_id& ext_prim_id = "",
                     const padding& output_padding = padding())
        : primitive_base(id, {input, seq_lengths}, ext_prim_id, {output_padding}), seq_axis(seq_axis), batch_axis(batch_axis) {
        const int32_t number_of_dims = 4;

        int32_t batch_a = batch_axis;
        int32_t seq_a = seq_axis;

        if (batch_a < 0)
            batch_a += number_of_dims;

        if (seq_a < 0)
            seq_a += number_of_dims;

        if (batch_a == seq_a)
            throw std::runtime_error("Batch axis and sequence axis should not be equal\n");

        if (batch_a < 0 || batch_a >= number_of_dims)
            throw std::runtime_error("Incorrect batch axis value! Actual axis is" + std::to_string(batch_a));

        if (seq_a < 0 || seq_a >= number_of_dims)
            throw std::runtime_error("Incorrect sequence axis value! Actual axis is" + std::to_string(seq_a));
    }

    /// @brief The axis which is partially reversed.
    int32_t seq_axis;
    /// @brief The axis along which reversal is performed.
    int32_t batch_axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
