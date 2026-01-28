// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "yaml_metadata.hpp"

#include <fstream>

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

YAMLMetadataReader::YAMLMetadataReader(const std::string& yaml_path) : m_yaml_path(yaml_path) {
    try {
        m_yaml = YAML::LoadFile(yaml_path);
        parse_yaml_metadata();
    } catch (const YAML::BadFile& e) {
        FRONT_END_GENERAL_CHECK(false, "Failed to open YAML metadata file: '", yaml_path, "'");
    } catch (const YAML::ParserException& e) {
        FRONT_END_GENERAL_CHECK(false,
                                "Failed to parse YAML metadata file '",
                                yaml_path,
                                "': ",
                                e.what(),
                                " at line ",
                                e.mark.line,
                                ", column ",
                                e.mark.column);
    } catch (const YAML::Exception& e) {
        FRONT_END_GENERAL_CHECK(false, "Error loading YAML metadata file '", yaml_path, "': ", e.what());
    }
}

void YAMLMetadataReader::parse_yaml_metadata() {
    try {
        // Parse the YAML structure from PP-OCRv5
        // Example structure:
        // model_type: ocr
        // model_specs:
        //   architecture: PP-OCRv5
        //   det_arch: DBNet
        //   rec_arch: SVTR

        // Parse model_type (optional)
        if (m_yaml["model_type"]) {
            m_metadata["model_type"] = m_yaml["model_type"].as<std::string>();
        }

        // Parse model_specs (optional)
        if (m_yaml["model_specs"]) {
            const auto& specs = m_yaml["model_specs"];
            if (specs.IsMap()) {
                for (const auto& spec : specs) {
                    std::string key = spec.first.as<std::string>();
                    std::string value = spec.second.as<std::string>();
                    m_metadata[key] = value;
                }
            }
        }

        // Add any additional top-level fields
        for (const auto& node : m_yaml) {
            std::string key = node.first.as<std::string>();
            if (key != "model_type" && key != "model_specs") {
                if (node.second.IsScalar()) {
                    m_metadata[key] = node.second.as<std::string>();
                }
            }
        }

    } catch (const YAML::Exception& e) {
        FRONT_END_GENERAL_CHECK(false, "Error parsing YAML metadata from '", m_yaml_path, "': ", e.what());
    }
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov