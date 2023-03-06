// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

// TODO: Remove OCL impl dependency here or move to OCL folder
#include "impls/ocl/kernel_selector_helper.h"

#include <vector>

namespace cldnn {

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
struct generic_layer : public primitive_base<generic_layer> {
    CLDNN_DECLARE_PRIMITIVE(generic_layer)

    /// @brief Constructs generic_layer primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    generic_layer(const primitive_id& id,
                  const primitive_id& input,
                  const layout& output_layout,
                  const kernel_selector::generic_kernel_params& generic_params,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), output_layout(output_layout), generic_params(generic_params) {}

    /// @brief Requested memory layout.
    layout output_layout;
    const kernel_selector::generic_kernel_params generic_params;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
