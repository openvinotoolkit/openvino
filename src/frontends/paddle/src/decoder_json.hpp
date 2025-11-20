// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "openvino/frontend/paddle/decoder.hpp"
#include "operator.hpp"

namespace ov {
namespace frontend {
namespace paddle {

/**
 * @brief Decoder for PaddlePaddle JSON format models (PP-OCRv5 and later)
 *
 * This decoder parses PaddlePaddle models stored in JSON format (inference.json)
 * introduced in PaddlePaddle 3.0 (PP-OCRv5). It implements the DecoderBase interface
 * to provide model structure information to the OpenVINO frontend.
 *
 * The JSON format contains:
 * - ops: Array of operators with type, inputs, outputs, and attributes
 * - vars: Array of variables with name, shape, and dtype information
 * - version: Model format version
 */
class DecoderJSON : public DecoderBase {
public:
    /**
     * @brief Construct a new DecoderJSON object from a JSON file path
     * @param json_path Path to the inference.json file
     * @throws ov::frontend::GeneralFailure if file cannot be opened or JSON is malformed
     */
    explicit DecoderJSON(const std::string& json_path);

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

    /**
     * @brief Get the IR version string from the JSON model
     * @return IR version string (e.g., "2.3" for PP-OCRv5)
     */
    std::string get_ir_version() const;

    /**
     * @brief Get the number of operators in the model
     * @return Number of operators
     */
    size_t get_op_size() const;

    /**
     * @brief Get operator at specified index
     * @param idx Operator index
     * @return Const reference to the Operator structure
     * @throws ov::frontend::GeneralFailure if index is out of range
     */
    const Operator& get_op(size_t idx) const;

    /**
     * @brief Get the raw JSON model data
     * @return Const reference to the parsed JSON object
     */
    const nlohmann::json& get_model_json() const {
        return m_model_json;
    }

private:
    nlohmann::json m_model_json;        ///< Parsed JSON model data
    std::vector<Operator> m_operators;  ///< Parsed operators from the model

    // Type and shape information extracted from vars array
    std::map<std::string, ov::element::Type> m_var_types;  ///< Variable name to element type mapping
    std::map<std::string, ov::PartialShape> m_var_shapes;  ///< Variable name to shape mapping

    /**
     * @brief Parse the JSON model and extract operators
     * @throws ov::frontend::GeneralFailure if JSON structure is invalid
     */
    void parse_json_model();

    /**
     * @brief Parse variable type and shape information from vars array
     *
     * Extracts dtype and shape information for each variable to enable
     * proper type inference. This method does not throw on failure.
     */
    void parse_vars_info();

    /**
     * @brief Parse PaddlePaddle dtype string to OpenVINO element type
     * @param dtype_str Dtype string from JSON (e.g., "float32", "int64")
     * @return Corresponding OpenVINO element type, defaults to f32 if unknown
     */
    ov::element::Type parse_dtype(const std::string& dtype_str) const;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov