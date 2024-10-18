// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/decoder.hpp"
#include "tf_framework_node.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class DecoderFake : public ov::frontend::DecoderBase {
public:
    explicit DecoderFake() {}

    ov::Any get_attribute(const std::string& name) const override {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: the get_attribute method of the fake node decoder is invoked.");
    }

    size_t get_input_size() const override {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: the get_input_size method of the fake node decoder is invoked.");
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: the get_input_node method of the fake node decoder is invoked.");
    }

    const std::string& get_op_type() const override {
        // this method must not throw an exception since it is used by TF FE FrameworkNode constructor
        return op_type;
    }

    const std::string& get_op_name() const override {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: the get_op_name method of the fake node decoder is invoked.");
    }

private:
    const std::string op_type = "fake";
};

class InternalOperation : public ov::frontend::tensorflow::FrameworkNode {
protected:
    InternalOperation(const std::shared_ptr<DecoderBase>& decoder,
                      const OutputVector& inputs,
                      size_t num_outputs,
                      const std::string& no_conversion_reason)
        : ov::frontend::tensorflow::FrameworkNode(decoder != nullptr ? decoder : std::make_shared<DecoderFake>(),
                                                  inputs,
                                                  num_outputs),
          m_no_conversion_reason(no_conversion_reason) {}

public:
    OPENVINO_OP("InternalOperation", "util", ov::frontend::tensorflow::FrameworkNode);
    // get a reason why some operation is unable to convert to OpenVINO opset
    // we store this information for InternalOperation to elaborate the reason
    // for cases such as Constant node of string type
    std::string get_no_conversion_reason() const {
        return m_no_conversion_reason;
    }

private:
    std::string m_no_conversion_reason;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
