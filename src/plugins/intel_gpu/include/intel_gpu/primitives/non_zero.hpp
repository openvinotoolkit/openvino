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
                  const input_info& data,
                  const padding& output_padding = padding())
        : primitive_base(id, {data}, {output_padding}) {}
};

struct gather_nonzero : public primitive_base<gather_nonzero> {
    CLDNN_DECLARE_PRIMITIVE(gather_nonzero)

    /// @brief Constructs gather_nonzero primitive.
    /// @param id This primitive id.
    /// @param data Input data primitive id.
    /// @param output_shape Output shape [rank of data, number of nonzero elements]
    gather_nonzero(const primitive_id& id,
                   const input_info& data,
                   const input_info& output_shape,
                   const padding& output_padding = padding())
        : primitive_base(id, {data, output_shape}, {output_padding}) {}
};

}  // namespace cldnn
