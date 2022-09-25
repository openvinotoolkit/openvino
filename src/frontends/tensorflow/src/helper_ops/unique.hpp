// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class Unique : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("Unique", "ov::frontend::tensorflow::util", ov::frontend::tensorflow::InternalOperation);

    Unique(const Output<Node>& input_values,
           ov::element::Type output_indices_type,
           const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder, OutputVector{input_values}, 2),
          out_idx(output_indices_type) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Unique finds unique elements in a 1-D tensor.
        // Inputs:
        // 0) 1D tensor of values
        // Outputs:
        // 0) 1D tensor of unique elements
        // 1) 1D tensor of indices of the unique elements in the input
        set_output_type(0, get_input_element_type(0), ov::PartialShape({ov::Dimension::dynamic()}));
        set_output_type(1, out_idx, ov::PartialShape({ov::Dimension::dynamic()}));
    }

private:
    ov::element::Type out_idx;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
