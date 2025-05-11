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

// Internal operation for Enter that marks entry point for data going to Loop in the graph
// It is used along with Exit operation
class Enter : public InternalOperation {
public:
    OPENVINO_OP("Enter", "ov::frontend::tensorflow", InternalOperation);

    Enter(const Output<Node>& data,
          const std::string frame_name,
          const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{data}, 1, "Enter"),
          m_frame_name(frame_name) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data_type = get_input_element_type(0);
        auto data_shape = get_input_partial_shape(0);

        set_output_type(0, data_type, data_shape);
    }

    std::string get_frame_name() const {
        return m_frame_name;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 1,
                                      "[TensorFlow Frontend] internal error: Enter expects one input");
        auto enter_node = std::make_shared<Enter>(inputs[0], m_frame_name, m_decoder);
        enter_node->set_attrs(get_attrs());
        return enter_node;
    }

private:
    std::string m_frame_name;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
