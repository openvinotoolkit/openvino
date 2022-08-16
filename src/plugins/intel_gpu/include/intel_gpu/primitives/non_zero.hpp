// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

struct count_nonzero : public primitive_base<count_nonzero> {
    CLDNN_DECLARE_PRIMITIVE(count_nonzero)

    /// @brief Constructs count_nonzero primitive.
    /// @param id This primitive id.
    /// @param data Input data primitive id.
    count_nonzero(const primitive_id& id,
                  const primitive_id& data,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {data}, ext_prim_id, output_padding) {}
};

struct gather_nonzero : public primitive_base<gather_nonzero> {
    CLDNN_DECLARE_PRIMITIVE(gather_nonzero)

    /// @brief Constructs gather_nonzero primitive.
    /// @param id This primitive id.
    /// @param data Input data primitive id.
    /// @param output_shape Output shape [rank of data, number of nonzero elements]
    gather_nonzero(const primitive_id& id,
                   const primitive_id& data,
                   const primitive_id& output_shape,
                   const primitive_id& ext_prim_id = "",
                   const padding& output_padding = padding())
        : primitive_base(id, {data, output_shape}, ext_prim_id, output_padding) {}
};

}  // namespace cldnn
