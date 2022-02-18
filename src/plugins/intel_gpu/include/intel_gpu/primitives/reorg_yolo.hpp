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

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
/// @par Where:
struct reorg_yolo : public primitive_base<reorg_yolo> {
    CLDNN_DECLARE_PRIMITIVE(reorg_yolo)

    /// @brief Constructs region_yolo primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    reorg_yolo(const primitive_id& id,
               const primitive_id& input,
               const uint32_t stride,
               const primitive_id& ext_prim_id = "",
               const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding), stride(stride) {}

    /// @brief Defines a scope of a reorg yolo normalization
    /// @details
    /// Specific behaviour is determined by these parameters, as follows:
    uint32_t stride;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
#pragma once
