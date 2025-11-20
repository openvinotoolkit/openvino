// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_input_model_imp.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>

#include "decoder_json.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "yaml_metadata.hpp"

namespace ov {
namespace frontend {
namespace paddle {

JSONInputModel::JSONInputModel(const std::string& json_path,
                               const std::string& yml_path,
                               const std::string& params_path,
                               const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_telemetry(telemetry) {
    load_model_structure(json_path);
    if (!yml_path.empty()) {
        load_metadata(yml_path);
    }
    if (!params_path.empty()) {
        load_weights(params_path);
    }
}

void JSONInputModel::load_model_structure(const std::string& json_path) {
    std::ifstream json_file(json_path);
    FRONT_END_GENERAL_CHECK(json_file.is_open(), "Cannot open JSON model file: ", json_path);

    json_file >> m_model_json;

    // Parse model structure from JSON
    // The PP-OCRv5 JSON format looks like:
    // {
    //   "ops": [
    //     {
    //       "type": "conv2d",
    //       "inputs": {...},
    //       "outputs": {...},
    //       "attrs": {...}
    //     }
    //   ],
    //   "vars": [...],
    //   "version": "2.3"
    // }

    try {
        // First create all tensor places
        if (m_model_json.contains("vars")) {
            for (const auto& var : m_model_json["vars"]) {
                auto var_place =
                    std::make_shared<JSONTensorPlace>(static_cast<const ov::frontend::InputModel&>(*this), var);
                m_var_places[var["name"].get<std::string>()] = var_place;
            }
        }

        // Parse operators and connect inputs/outputs
        for (const auto& op : m_model_json["ops"]) {
            FRONT_END_GENERAL_CHECK(op.contains("type"), "Invalid operator format: missing 'type' field");
            const auto& op_type = op["type"].get<std::string>();

            // Special handling for input/output ops
            if (op_type == "feed") {
                // Handle input op
                FRONT_END_GENERAL_CHECK(op.contains("outputs"), "Feed operator missing outputs");
                for (const auto& [var_name, output_list] : op["outputs"].items()) {
                    if (output_list.is_string()) {
                        auto var_name_str = output_list.get<std::string>();
                        auto var_place = m_var_places[var_name_str];
                        if (var_place) {
                            m_inputs.push_back(std::static_pointer_cast<Place>(var_place));
                        }
                    }
                }
            } else if (op_type == "fetch") {
                // Handle output op
                FRONT_END_GENERAL_CHECK(op.contains("inputs"), "Fetch operator missing inputs");
                for (const auto& [var_name, input_list] : op["inputs"].items()) {
                    if (input_list.is_string()) {
                        auto var_name_str = input_list.get<std::string>();
                        auto var_place = m_var_places[var_name_str];
                        if (var_place) {
                            m_outputs.push_back(std::static_pointer_cast<Place>(var_place));
                        }
                    }
                }
            }
        }

    } catch (const nlohmann::json::exception& e) {
        FRONT_END_THROW("Error parsing JSON model: " + std::string(e.what()));
    }
}

void JSONInputModel::load_metadata(const std::string& yml_path) {
    try {
        YAMLMetadataReader yaml_reader(yml_path);
        const auto& metadata = yaml_reader.get_metadata();

        // Validate model type (optional - just log warning if not OCR)
        if (yaml_reader.has_field("model_type")) {
            const auto& model_type = yaml_reader.get_field("model_type");
            if (model_type != "ocr") {
                // Log warning but don't fail - JSON format may be used for other model types
                // OPENVINO_WARN << "Model type is '" << model_type << "', expected 'ocr'";
            }
        }

        // Check architecture (optional - just log if not PP-OCRv5)
        if (yaml_reader.has_field("architecture")) {
            const auto& arch = yaml_reader.get_field("architecture");
            if (arch != "PP-OCRv5") {
                // Log info but don't fail - may support other architectures in future
                // OPENVINO_INFO << "Model architecture is '" << arch << "'";
            }
        }

        // Store all metadata for potential future use
        m_metadata = metadata;

    } catch (const std::exception& e) {
        // Don't fail if metadata loading fails - it's supplementary information
        // The JSON model itself should be sufficient for conversion
        // OPENVINO_WARN << "Failed to load YAML metadata: " << e.what();
    }
}

void JSONInputModel::load_weights(const std::string& params_path) {
    // TODO: Implement weight loading from .pdiparams file
    // The .pdiparams format is the same for both JSON and protobuf models,
    // so we should be able to reuse the existing weight loading logic.
    //
    // For now, this is a stub. Weight loading will need to:
    // 1. Parse the binary .pdiparams file format
    // 2. Map weight names to tensor places
    // 3. Load weight data into the appropriate tensors
    //
    // The weight loading implementation should follow the pattern used in
    // the protobuf-based InputModel (input_model.cpp)

    std::ifstream params_file(params_path, std::ios::binary);
    if (!params_file.is_open()) {
        // Weights file is optional for some models (e.g., models with no learnable parameters)
        return;
    }

    // Weight loading implementation needed here
    // For now, just verify the file exists and is readable
    params_file.close();
}

std::vector<Place::Ptr> JSONInputModel::get_places() const {
    return collect_all_places();
}

std::vector<Place::Ptr> JSONInputModel::collect_all_places() const {
    std::vector<Place::Ptr> all_places;
    all_places.reserve(m_var_places.size());

    // Add all tensor places
    for (const auto& [name, place] : m_var_places) {
        all_places.push_back(std::static_pointer_cast<Place>(place));
    }

    return all_places;
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov