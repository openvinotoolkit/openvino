// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_op.hpp"
#include "openvino/frontend/decoder.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class AlignTypes : public InternalOperation {
public:
    AlignTypes(const Output<Node>& lhs, const Output<Node>& rhs, bool align_scalars)
        : InternalOperation("ov::align_types",
                            {lhs, rhs},
                            2,
                            "This is internal operation for type alignment and should be removed "
                            "at normalization step. It can't be removed if types can't be resolved."),
          m_align_scalars(align_scalars) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto lhs = input_value(0);
        auto rhs = input_value(1);
        auto out_type = infer_types(lhs, rhs, m_align_scalars);
        set_output_type(0, out_type, get_input_partial_shape(0));
        set_output_type(1, out_type, get_input_partial_shape(1));
    }

private:
    const bool m_align_scalars;
};
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
