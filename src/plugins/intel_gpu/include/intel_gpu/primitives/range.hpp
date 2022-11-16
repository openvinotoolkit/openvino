// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct range: public primitive_base<range> {
    CLDNN_DECLARE_PRIMITIVE(range)

    /// @brief Constructs range primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitive id vector.
    /// @param output_layout requested range output layout
    range(const primitive_id &id,
          const std::vector<primitive_id> &input,
          const layout &output_layout)
        : primitive_base{ id, input, output_layout.data_padding, output_layout.data_type },
          output_layout { output_layout } { }

    /// @brief requested range output layout
    layout output_layout;
};
}  // namespace cldnn
