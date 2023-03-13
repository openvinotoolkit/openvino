// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/tensorflow/decoder.hpp"

namespace tensorflow {
class OpDef_ArgDef;
}  // namespace tensorflow

namespace ov {
namespace frontend {
namespace tensorflow {

class DecoderArgDef : public ov::frontend::tensorflow::DecoderBase {
public:
    explicit DecoderArgDef(const ::tensorflow::OpDef_ArgDef* arg_def, const std::string& op_type)
        : m_arg_def(arg_def),
          m_op_type(op_type) {}

    explicit DecoderArgDef(const ::tensorflow::OpDef_ArgDef* arg_def,
                           const std::string& op_type,
                           const std::string& producer_name)
        : m_arg_def(arg_def),
          m_op_type(op_type),
          m_producer_name(producer_name) {}

    ov::Any get_attribute(const std::string& name) const override;

    size_t get_input_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index,
                        const OpTypeByName& op_type_by_name) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

private:
    const ::tensorflow::OpDef_ArgDef* m_arg_def;
    const std::string m_op_type;
    const std::string m_producer_name;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
