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

class YAMLMetadataReader {
public:
    explicit YAMLMetadataReader(const std::string& yaml_path);

    // Get metadata information
    std::map<std::string, std::string> get_metadata() const {
        return m_metadata;
    }

private:
    YAML::Node m_yaml;
    std::map<std::string, std::string> m_metadata;

    void parse_yaml_metadata();
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov