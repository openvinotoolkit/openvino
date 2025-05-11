// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class Rfft2d : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("Rfft2d", "ov::frontend::tensorflow_lite::util", ov::frontend::tensorflow::InternalOperation);

    Rfft2d(const Output<Node>& data,
           const Output<Node>& fft_length,
           const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder, OutputVector{data, fft_length}, 1, "Rfft2d") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data_rank = get_input_partial_shape(0).rank();
        NODE_VALIDATION_CHECK(this, data_rank.is_dynamic() || data_rank.get_length() >= 2);
        auto length_rank = get_input_partial_shape(1).rank();
        NODE_VALIDATION_CHECK(this, length_rank.compatible(1));
        set_output_type(0, element::dynamic, PartialShape::dynamic(get_input_partial_shape(0).rank()));
    }
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
