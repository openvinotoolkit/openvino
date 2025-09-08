// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "internal_operation.hpp"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for Exit that marks exit point for data going from Loop in the graph
// It is used along with Enter operation
class Exit : public InternalOperation {
public:
    OPENVINO_OP("Exit", "ov::frontend::tensorflow", InternalOperation);

    Exit(const Output<Node>& data, const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{data}, 1, "Exit") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data_type = get_input_element_type(0);
        auto data_shape = get_input_partial_shape(0);

        set_output_type(0, data_type, data_shape);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 1,
                                      "[TensorFlow Frontend] internal error: Exit expects one input");
        auto exit_node = std::make_shared<Exit>(inputs[0], m_decoder);
        exit_node->set_attrs(get_attrs());
        return exit_node;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
