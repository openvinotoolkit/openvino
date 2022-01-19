// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct range: public primitive_base<range> {
    CLDNN_DECLARE_PRIMITIVE(range)

    range(const primitive_id &id, const std::vector<primitive_id> &input, const layout &output_layout,
          const primitive_id &ext_prim_id = { }) :
        primitive_base { id, input, ext_prim_id, output_layout.data_padding }, output_layout { output_layout } {
    }
    layout output_layout;
};
}  // namespace cldnn
