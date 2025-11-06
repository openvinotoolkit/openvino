// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_json.hpp"

#include <fstream>

#include "openvino/core/except.hpp"
#include "openvino/frontend/paddle/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

DecoderJSON::DecoderJSON(const std::string& json_path) {
    std::ifstream json_file(json_path);
    FRONT_END_GENERAL_CHECK(json_file.is_open(), "Cannot open JSON model file: ", json_path);
    json_file >> m_model_json;
    parse_json_model();
}

void DecoderJSON::parse_json_model() {
    if (!m_model_json.contains("ops") || !m_model_json["ops"].is_array()) {
        FRONT_END_THROW("Invalid model format: missing 'ops' array");
    }

    try {
        for (const auto& op : m_model_json["ops"]) {
            if (!op.contains("type")) {
                FRONT_END_THROW("Invalid operator format: missing 'type' field");
            }

            Operator paddle_op;
            paddle_op.type = op["type"].get<std::string>();

            // Parse inputs
            if (op.contains("inputs")) {
                for (const auto& [var_name, input_list] : op["inputs"].items()) {
                    if (input_list.is_string()) {
                        paddle_op.inputs.push_back(input_list.get<std::string>());
                    } else if (input_list.is_array()) {
                        for (const auto& input_name : input_list) {
                            paddle_op.inputs.push_back(input_name.get<std::string>());
                        }
                    }
                }
            }

            // Parse outputs
            if (op.contains("outputs")) {
                for (const auto& [var_name, output_list] : op["outputs"].items()) {
                    if (output_list.is_string()) {
                        paddle_op.outputs.push_back(output_list.get<std::string>());
                    } else if (output_list.is_array()) {
                        for (const auto& output_name : output_list) {
                            paddle_op.outputs.push_back(output_name.get<std::string>());
                        }
                    }
                }
            }

            // Parse attributes
            if (op.contains("attrs")) {
                for (const auto& [key, value] : op["attrs"].items()) {
                    paddle_op.attributes[key] = value;
                }
            }

            m_operators.push_back(paddle_op);
        }
    } catch (const nlohmann::json::exception& e) {
        FRONT_END_THROW("Error parsing JSON model: " + std::string(e.what()));
    }
}

int64_t DecoderJSON::get_version() const {
    if (m_model_json.contains("version")) {
        const auto& version = m_model_json["version"];
        if (version.is_string()) {
            return std::stoll(version.get<std::string>());
        }
        return version.get<int64_t>();
    }
    return -1;  // Default version or error value
}

std::string DecoderJSON::get_ir_version() const {
    if (m_model_json.contains("ir_version")) {
        return m_model_json["ir_version"].get<std::string>();
    }
    return "2.3";  // Default PP-OCRv5 version
}

size_t DecoderJSON::get_op_size() const {
    return m_operators.size();
}

const Operator& DecoderJSON::get_op(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < m_operators.size(), "Operator index out of range");
    return m_operators[idx];
}

ov::Any DecoderJSON::get_attribute(const std::string& name) const {
    // Look through all operators to find one with matching attribute
    for (const auto& op : m_operators) {
        auto it = op.attributes.find(name);
        if (it != op.attributes.end()) {
            return it->second;
        }
    }
    return {};
}

ov::Any DecoderJSON::convert_attribute(const ov::Any& data, const std::type_info& type_info) const {
    if (data.empty()) {
        return {};
    }

    // Handle basic types
    if (type_info == typeid(int64_t)) {
        return data.as<int64_t>();
    } else if (type_info == typeid(float)) {
        return data.as<float>();
    } else if (type_info == typeid(bool)) {
        return data.as<bool>();
    } else if (type_info == typeid(std::string)) {
        return data.as<std::string>();
    }

    // Return as-is for unsupported types
    return data;
}

std::vector<OutPortName> DecoderJSON::get_output_names() const {
    std::vector<OutPortName> result;
    for (const auto& op : m_operators) {
        result.insert(result.end(), op.outputs.begin(), op.outputs.end());
    }
    return result;
}

std::vector<TensorName> DecoderJSON::get_output_var_names(const std::string& var_name) const {
    std::vector<TensorName> result;
    for (const auto& op : m_operators) {
        for (const auto& out : op.outputs) {
            if (out == var_name) {
                result.push_back(out);
            }
        }
    }
    return result;
}

std::vector<TensorName> DecoderJSON::get_input_var_names(const std::string& var_name) const {
    std::vector<TensorName> result;
    for (const auto& op : m_operators) {
        for (const auto& in : op.inputs) {
            if (in == var_name) {
                result.push_back(in);
            }
        }
    }
    return result;
}

size_t DecoderJSON::get_output_size() const {
    size_t total = 0;
    for (const auto& op : m_operators) {
        total += op.outputs.size();
    }
    return total;
}

size_t DecoderJSON::get_output_size(const std::string& port_name) const {
    for (const auto& op : m_operators) {
        auto count = std::count(op.outputs.begin(), op.outputs.end(), port_name);
        if (count > 0) {
            return count;
        }
    }
    return 0;
}

ov::element::Type DecoderJSON::get_out_port_type(const std::string& port_name) const {
    // For now, assume all ports are float32 as PP-OCRv5 primarily uses float32
    return ov::element::f32;
}

std::vector<std::pair<ov::element::Type, ov::PartialShape>> DecoderJSON::get_output_port_infos(
    const std::string& port_name) const {
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> result;
    if (get_output_size(port_name) > 0) {
        // For now, assume all ports are float32 with dynamic shape as PP-OCRv5 uses dynamic input shapes
        result.push_back({ov::element::f32, ov::PartialShape::dynamic()});
    }
    return result;
}

std::string DecoderJSON::get_op_type() const {
    if (!m_operators.empty()) {
        return m_operators[0].type;
    }
    return "";
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov