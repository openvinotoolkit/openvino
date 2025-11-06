// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "yaml_metadata.hpp"

#include <fstream>

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace paddle {

YAMLMetadataReader::YAMLMetadataReader(const std::string& yaml_path) {
    try {
        m_yaml = YAML::LoadFile(yaml_path);
        parse_yaml_metadata();
    } catch (const YAML::Exception& e) {
        FRONT_END_THROW("Error loading YAML metadata file: " + std::string(e.what()));
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
        //   ...

        if (m_yaml["model_type"]) {
            m_metadata["model_type"] = m_yaml["model_type"].as<std::string>();
        }

        if (m_yaml["model_specs"]) {
            const auto& specs = m_yaml["model_specs"];
            for (const auto& spec : specs) {
                m_metadata[spec.first.as<std::string>()] = spec.second.as<std::string>();
            }
        }

        // Add any additional metadata parsing as needed

    } catch (const YAML::Exception& e) {
        FRONT_END_THROW("Error parsing YAML metadata: " + std::string(e.what()));
    }
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov