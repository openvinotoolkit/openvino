// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct range: public primitive_base<range> {
    CLDNN_DECLARE_PRIMITIVE(range)

    range(const primitive_id& id,
          const std::vector<primitive_id>& input,
          const tensor& output_shape,
          const data_types output_dt = data_types::f32,
          const padding& output_padding = padding())
        : primitive_base(id, input, output_padding, output_dt),
          output_shape(output_shape) {}

    range(const primitive_id& id,
          const std::vector<primitive_id>& input,
          const data_types output_dt = data_types::f32,
          const padding& output_padding = padding())
        : primitive_base(id, input, output_padding, output_dt),
          output_shape(tensor()) {}

    tensor output_shape;
};
}  // namespace cldnn
