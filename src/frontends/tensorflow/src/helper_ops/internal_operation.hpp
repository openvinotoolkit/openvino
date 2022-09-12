// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "op_table.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "tf_framework_node.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class DecoderFake : public ov::frontend::tensorflow::DecoderBase {
public:
    explicit DecoderFake() {}

    ov::Any get_attribute(const std::string& name) const override {
        return ov::Any("fake");
    }

    size_t get_input_size() const override {
        return 0;
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override {}

    const std::string& get_op_type() const override {
        return std::string("fake");
    }

    const std::string& get_op_name() const override {
        return std::string("fake");
    }
};

class InternalOperation : public ov::frontend::tensorflow::FrameworkNode {
public:
    InternalOperation(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs, size_t num_outputs)
        : ov::frontend::tensorflow::FrameworkNode(decoder, inputs, num_outputs) {}
    InternalOperation(const OutputVector& inputs, size_t num_outputs)
        : ov::frontend::tensorflow::FrameworkNode(std::make_shared<DecoderFake>(), inputs, num_outputs) {}
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
