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
struct region_yolo : public primitive_base<region_yolo> {
    CLDNN_DECLARE_PRIMITIVE(region_yolo)

    /// @brief Constructs region_yolo primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    region_yolo(const primitive_id& id,
                const primitive_id& input,
                const uint32_t coords,
                const uint32_t classes,
                const uint32_t num,
                const uint32_t mask_size = 0,
                const bool do_softmax = true,
                const primitive_id& ext_prim_id = "",
                const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          coords(coords),
          classes(classes),
          num(num),
          mask_size(mask_size),
          do_softmax(do_softmax) {}

    /// @brief Defines a scope of a region yolo normalization
    /// @details
    /// Specific behaviour is determined by these parameters, as follows:
    uint32_t coords;
    uint32_t classes;
    uint32_t num;
    uint32_t mask_size;
    bool do_softmax;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
#pragma once
