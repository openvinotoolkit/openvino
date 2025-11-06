// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "openvino/frontend/paddle/decoder.hpp"
#include "operator.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class DecoderJSON : public DecoderBase {
public:
    DecoderJSON(const std::string& json_path);
    virtual ~DecoderJSON() = default;

    // Override required DecoderBase interface methods
    int64_t get_version() const override;
    ov::Any get_attribute(const std::string& name) const override;
    ov::Any convert_attribute(const ov::Any& data, const std::type_info& type_info) const override;
    std::vector<OutPortName> get_output_names() const override;
    std::vector<TensorName> get_output_var_names(const std::string& var_name) const override;
    std::vector<TensorName> get_input_var_names(const std::string& var_name) const override;
    size_t get_output_size() const override;
    size_t get_output_size(const std::string& port_name) const override;
    ov::element::Type get_out_port_type(const std::string& port_name) const override;
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> get_output_port_infos(
        const std::string& port_name) const override;
    std::string get_op_type() const override;

    // Additional methods for JSON model handling
    std::string get_ir_version() const;
    size_t get_op_size() const;
    const Operator& get_op(size_t idx) const;

private:
    nlohmann::json m_model_json;
    std::vector<Operator> m_operators;

    void parse_json_model();
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov