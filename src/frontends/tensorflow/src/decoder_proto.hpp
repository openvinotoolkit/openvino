// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

class DecoderProto : public ov::frontend::tensorflow::DecoderBase {
public:
    explicit DecoderProto(const ::tensorflow::NodeDef* node_def) : m_node_def(node_def) {}

    ov::Any get_attribute(const std::string& name) const override;

    ov::Any get_native_attribute(const std::string& name) const override;

    size_t get_input_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

private:
    std::vector<::tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const ::tensorflow::NodeDef* m_node_def;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
