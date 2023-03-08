// Copyright (C) 2018-2023 Intel Corporation
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
                        size_t& producer_output_port_index) const override {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      "Internal error: the get_input_node method of the fake node decoder is invoked.");
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index,
                        const OpTypeByName& op_type_by_name) const override {
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
public:
    InternalOperation(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs, size_t num_outputs)
        : ov::frontend::tensorflow::FrameworkNode(decoder != nullptr ? decoder : std::make_shared<DecoderFake>(),
                                                  inputs,
                                                  num_outputs) {}
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
