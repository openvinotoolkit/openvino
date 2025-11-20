// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "tensor_place.hpp"

namespace ov {
namespace frontend {
namespace paddle {

/**
 * @brief Input model implementation for PaddlePaddle JSON format (PP-OCRv5)
 *
 * This class handles loading and parsing of PaddlePaddle 3.0 models stored in JSON format.
 * It manages the model structure, metadata, and weights.
 */
class JSONInputModel : public ov::frontend::InputModel {
public:
    using Ptr = std::shared_ptr<JSONInputModel>;

    /**
     * @brief Construct a JSONInputModel from file paths
     * @param json_path Path to inference.json file
     * @param yml_path Path to inference.yml metadata file (optional)
     * @param params_path Path to inference.pdiparams weights file (optional)
     * @param telemetry Telemetry extension for tracking (optional)
     */
    JSONInputModel(const std::string& json_path,
                   const std::string& yml_path,
                   const std::string& params_path,
                   const std::shared_ptr<TelemetryExtension>& telemetry = nullptr);

    // Implement required InputModel interface
    std::vector<Place::Ptr> get_inputs() const override {
        return m_inputs;
    }
    std::vector<Place::Ptr> get_outputs() const override {
        return m_outputs;
    }
    std::vector<Place::Ptr> get_places() const;

private:
    /**
     * @brief Load and parse the JSON model structure
     * @param json_path Path to inference.json file
     */
    void load_model_structure(const std::string& json_path);

    /**
     * @brief Load YAML metadata (optional)
     * @param yml_path Path to inference.yml file
     */
    void load_metadata(const std::string& yml_path);

    /**
     * @brief Load model weights from .pdiparams file
     * @param params_path Path to inference.pdiparams file
     */
    void load_weights(const std::string& params_path);

    /**
     * @brief Collect all places for get_places()
     * @return Vector of all tensor places
     */
    std::vector<Place::Ptr> collect_all_places() const;

    nlohmann::json m_model_json;                                           ///< Parsed JSON model data
    std::map<std::string, std::shared_ptr<JSONTensorPlace>> m_var_places;  ///< Tensor places by name
    std::vector<Place::Ptr> m_inputs;                                      ///< Model input places
    std::vector<Place::Ptr> m_outputs;                                     ///< Model output places
    std::map<std::string, std::string> m_metadata;                         ///< YAML metadata
    std::shared_ptr<TelemetryExtension> m_telemetry;                       ///< Telemetry extension
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov