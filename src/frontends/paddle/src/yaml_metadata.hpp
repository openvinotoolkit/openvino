// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <yaml-cpp/yaml.h>

#include <map>
#include <string>

namespace ov {
namespace frontend {
namespace paddle {

/**
 * @brief Reader for PaddlePaddle YAML metadata files
 *
 * Parses YAML metadata files (inference.yml) that accompany PP-OCRv5 JSON models.
 * The metadata contains model type, architecture information, and other configuration.
 */
class YAMLMetadataReader {
public:
    /**
     * @brief Construct a new YAMLMetadataReader from a YAML file path
     * @param yaml_path Path to the inference.yml file
     * @throws ov::frontend::GeneralFailure if file cannot be opened or parsed
     */
    explicit YAMLMetadataReader(const std::string& yaml_path);

    /**
     * @brief Get all metadata as a key-value map
     * @return Map of metadata keys to values
     */
    std::map<std::string, std::string> get_metadata() const {
        return m_metadata;
    }

    /**
     * @brief Check if a metadata field exists
     * @param key Metadata field name
     * @return true if the field exists, false otherwise
     */
    bool has_field(const std::string& key) const {
        return m_metadata.find(key) != m_metadata.end();
    }

    /**
     * @brief Get a metadata field value
     * @param key Metadata field name
     * @param default_value Default value if field doesn't exist
     * @return Field value or default_value if not found
     */
    std::string get_field(const std::string& key, const std::string& default_value = "") const {
        auto it = m_metadata.find(key);
        return (it != m_metadata.end()) ? it->second : default_value;
    }

private:
    YAML::Node m_yaml;
    std::string m_yaml_path;  ///< Path to YAML file for error messages
    std::map<std::string, std::string> m_metadata;

    /**
     * @brief Parse YAML metadata into the metadata map
     * @throws ov::frontend::GeneralFailure if parsing fails
     */
    void parse_yaml_metadata();
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov