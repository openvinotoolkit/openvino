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

class ComplexAbs : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("ComplexAbs", "ov::frontend::tensorflow_lite::util", ov::frontend::tensorflow::InternalOperation);

    ComplexAbs(const Output<Node>& data, const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder, OutputVector{data}, 1, "ComplexAbs") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, element::dynamic, get_input_partial_shape(0));
    }
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
