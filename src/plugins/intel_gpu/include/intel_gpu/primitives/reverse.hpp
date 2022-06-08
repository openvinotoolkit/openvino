// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "primitive.hpp"

namespace cldnn {

enum class reverse_mode : uint32_t { index, mask };

struct reverse : public primitive_base<reverse> {
    CLDNN_DECLARE_PRIMITIVE(reverse)

    /// @brief Constructs reverse primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axes Axes to reverse primitive id.
    /// @param mode Axes interpretation mode (indices/mask).
    reverse(const primitive_id& id,
            const primitive_id& input,
            const primitive_id& axes,
            const reverse_mode mode,
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base{id, {input, axes}, ext_prim_id, output_padding},
          mode{mode} {}

    reverse_mode mode{reverse_mode::index};
};
}  // namespace cldnn
