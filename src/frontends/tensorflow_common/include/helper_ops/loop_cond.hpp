// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "internal_operation.hpp"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for Loop that represents the loop termination condition
// by the pivot switches of a loop
class LoopCond : public InternalOperation {
public:
    OPENVINO_OP("LoopCond", "ov::frontend::tensorflow", InternalOperation);

    LoopCond(const Output<Node>& input, const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{input}, 1, "LoopCond") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data_type = get_input_element_type(0);
        auto data_shape = get_input_partial_shape(0);

        set_output_type(0, data_type, data_shape);
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
