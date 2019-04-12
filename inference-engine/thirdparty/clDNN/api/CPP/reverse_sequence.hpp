/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "../C/reverse_sequence.h"
#include "primitive.hpp"

namespace  cldnn
{
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief
/// @details
struct reverse_sequence : public primitive_base<reverse_sequence, CLDNN_PRIMITIVE_DESC(reverse_sequence)>
{
    CLDNN_DECLARE_PRIMITIVE(reverse_sequence)

    /// @brief Constructs reverse_sequence primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param seq_lengths Sequence lengths primitive id.
    /// @param seq_axis The axis which is partially reversed.
    /// @param batch_axis The axis along which reversal is performed.
    reverse_sequence(
            const primitive_id& id,
            const primitive_id& input,
            const primitive_id& seq_lengths,
            const int32_t seq_axis,
            const int32_t batch_axis = 0,
            const padding& output_padding = padding()
    )
            : primitive_base(id, {input, seq_lengths}, output_padding)
            , seq_axis(seq_axis)
            , batch_axis(batch_axis)
    {
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

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{reverse_sequence}
    reverse_sequence(const dto* dto)
            : primitive_base(dto)
            , seq_axis(dto->seq_axis)
            , batch_axis(dto->batch_axis)
    {
    }

    /// @brief The axis which is partially reversed.
    int32_t seq_axis;
    /// @brief The axis along which reversal is performed.
    int32_t batch_axis;
protected:

    void update_dto(dto& dto) const override
    {
        dto.seq_axis = seq_axis;
        dto.batch_axis = batch_axis;
    }
};
/// @}
/// @}
/// @}
}
