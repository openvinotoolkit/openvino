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

    try {
        json_file >> m_model_json;
    } catch (const nlohmann::json::exception& e) {
        FRONT_END_GENERAL_CHECK(false, "Failed to parse JSON model file '", json_path, "': ", e.what());
    }

    parse_json_model();
    parse_vars_info();
}

void DecoderJSON::parse_json_model() {
    FRONT_END_GENERAL_CHECK(m_model_json.contains("ops") && m_model_json["ops"].is_array(),
                            "Invalid model format: missing or invalid 'ops' array");

    try {
        const auto& ops = m_model_json["ops"];
        m_operators.reserve(ops.size());  // Optimize memory allocation

        for (const auto& op : ops) {
            FRONT_END_GENERAL_CHECK(op.contains("type"), "Invalid operator format: missing 'type' field");

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

            // Parse attributes - store entire attrs object
            if (op.contains("attrs")) {
                paddle_op.attributes = op["attrs"];
            }

            m_operators.push_back(std::move(paddle_op));
        }
    } catch (const nlohmann::json::exception& e) {
        FRONT_END_GENERAL_CHECK(false, "Error parsing JSON model operators: ", e.what());
    }
}

void DecoderJSON::parse_vars_info() {
    // Parse variable type and shape information from vars array if present
    if (!m_model_json.contains("vars") || !m_model_json["vars"].is_array()) {
        // Vars array is optional in some JSON formats
        return;
    }

    try {
        for (const auto& var : m_model_json["vars"]) {
            if (!var.contains("name")) {
                continue;  // Skip vars without names
            }

            std::string var_name = var["name"].get<std::string>();

            // Parse dtype (element type)
            if (var.contains("dtype")) {
                const auto& dtype_str = var["dtype"].get<std::string>();
                ov::element::Type elem_type = parse_dtype(dtype_str);
                m_var_types[var_name] = elem_type;
            }

            // Parse shape
            if (var.contains("shape") && var["shape"].is_array()) {
                std::vector<ov::Dimension> dims;
                for (const auto& dim : var["shape"]) {
                    if (dim.is_number_integer()) {
                        int64_t dim_val = dim.get<int64_t>();
                        if (dim_val == -1) {
                            dims.push_back(ov::Dimension::dynamic());
                        } else {
                            dims.push_back(ov::Dimension(dim_val));
                        }
                    } else if (dim.is_string() && dim.get<std::string>() == "-1") {
                        dims.push_back(ov::Dimension::dynamic());
                    }
                }
                m_var_shapes[var_name] = ov::PartialShape(dims);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        // Don't fail if vars parsing fails, just log and continue
        // Type inference will fall back to defaults
    }
}

ov::element::Type DecoderJSON::parse_dtype(const std::string& dtype_str) const {
    // Map PaddlePaddle dtype strings to OpenVINO element types
    static const std::map<std::string, ov::element::Type> dtype_map = {
        {"float32", ov::element::f32},
        {"float16", ov::element::f16},
        {"float64", ov::element::f64},
        {"int8", ov::element::i8},
        {"int16", ov::element::i16},
        {"int32", ov::element::i32},
        {"int64", ov::element::i64},
        {"uint8", ov::element::u8},
        {"bool", ov::element::boolean},
        {"bfloat16", ov::element::bf16},
        // Alternative names
        {"fp32", ov::element::f32},
        {"fp16", ov::element::f16},
        {"fp64", ov::element::f64},
    };

    auto it = dtype_map.find(dtype_str);
    if (it != dtype_map.end()) {
        return it->second;
    }

    // Default to float32 if unknown type
    return ov::element::f32;
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
        if (op.attributes.contains(name)) {
            return op.attributes[name];
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
    // Reserve approximate capacity to avoid reallocations
    size_t total_outputs = 0;
    for (const auto& op : m_operators) {
        total_outputs += op.outputs.size();
    }
    result.reserve(total_outputs);

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
    // Try to find type information from vars map
    if (m_var_types.count(port_name) > 0) {
        return m_var_types.at(port_name);
    }

    // Default to float32 if type information not available
    // PP-OCRv5 primarily uses float32 for most tensors
    return ov::element::f32;
}

std::vector<std::pair<ov::element::Type, ov::PartialShape>> DecoderJSON::get_output_port_infos(
    const std::string& port_name) const {
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> result;
    if (get_output_size(port_name) > 0) {
        // Get type from vars map if available, otherwise default to float32
        ov::element::Type elem_type = ov::element::f32;
        if (m_var_types.count(port_name) > 0) {
            elem_type = m_var_types.at(port_name);
        }

        // Get shape from vars map if available, otherwise use dynamic shape
        ov::PartialShape shape = ov::PartialShape::dynamic();
        if (m_var_shapes.count(port_name) > 0) {
            shape = m_var_shapes.at(port_name);
        }

        result.push_back({elem_type, shape});
    }
    return result;
}

std::string DecoderJSON::get_op_type() const {
    FRONT_END_GENERAL_CHECK(!m_operators.empty(), "Cannot get operator type: no operators in model");
    return m_operators[0].type;
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov