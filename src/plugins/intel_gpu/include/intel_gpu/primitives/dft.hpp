// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {

enum class dft_kind {
    forward,
    inverse,
};

struct dft: public primitive_base<dft> {
    CLDNN_DECLARE_PRIMITIVE(dft)

    dft(const primitive_id &id, primitive_id &&data, std::vector<int64_t> &&axes, const layout &output_layout,
        dft_kind kind, const primitive_id &ext_prim_id = { }) :
            primitive_base { id, { move(data) }, ext_prim_id, output_layout.data_padding },
            axes { move(axes) },
            output_layout { output_layout },
            kind { kind } {
    }
    std::vector<int64_t> axes;
    layout output_layout;
    dft_kind kind;
};
}  // namespace cldnn
