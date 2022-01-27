// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
struct gather_nd : public primitive_base<gather_nd> {
    CLDNN_DECLARE_PRIMITIVE(gather_nd)

    /// @brief Constructs gather_nd primitive.
    ///
    /// @param id                   This primitive id.
    /// @param data                 Input data primitive id.
    /// @param indices              Input indexes primitive id.
    /// @param input_rank           Rank of input data.
    /// @param indices_rank         Rank of indices.
    /// @param batch_dims           batch_dims as an attribute of GatherND. Optional.
    /// @param batch_merged_output  batched output shape is merged as a dimention for v5.
    ///                             In case of output{3, 2, 4, 5} at batch_dims = 2, real output shape should be {6, 4, 5}.
    ///                             This should be false for v8.
    ///                             For batch_dims < 2, This doesn't have any meaning.
    gather_nd(const primitive_id& id,
              const primitive_id& data,
              const primitive_id& indices,
              const uint8_t input_rank,
              const uint8_t indices_rank,
              const uint8_t batch_dims = 0,
              const bool batch_merged_output = true,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
        : primitive_base(id, {data, indices}, ext_prim_id, output_padding),
                         input_rank(input_rank),
                         indices_rank(indices_rank),
                         batch_dims(batch_dims),
                         batch_merged_output(batch_merged_output) {}

    /// @brief GatherND input_rank
    uint8_t input_rank;

    /// @brief GatherND indices_rank
    uint8_t indices_rank;

    /// @brief GatherND batch_dims
    uint8_t batch_dims;

    /// @brief GatherND batch_merged_output
    bool batch_merged_output;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
