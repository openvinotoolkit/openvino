// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/tensorflow/decoder.hpp"

namespace tflite {
class Operator;
}  // namespace tflite

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class DecoderFlatBuffer : public ov::frontend::tensorflow::DecoderBase {
public:
    explicit DecoderFlatBuffer(const tflite::Operator* node_def, const std::string& type, const std::string& name) : m_node_def(node_def), m_type(type), m_name(name) {}

    ov::Any get_attribute(const std::string& name) const override;

    size_t get_input_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

private:
//    std::vector<::tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const tflite::Operator* m_node_def;
    std::string m_type, m_name;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
