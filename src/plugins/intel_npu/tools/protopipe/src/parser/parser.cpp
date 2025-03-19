//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parser/parser.hpp"
#include "parser/config.hpp"

#include "utils/error.hpp"

#include <yaml-cpp/yaml.h>

ScenarioParser::ScenarioParser(const std::string& filepath): m_filepath(filepath) {
}

Config ScenarioParser::parseScenarios(const ReplaceBy& replace_by) {
    const auto root = YAML::LoadFile(m_filepath);
    // TODO: Extend to any other config syntax
    return parseConfig(root, replace_by);
}
