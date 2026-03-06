//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parser/parser.hpp"
#include "parser/config.hpp"
#include "parser/config_node.hpp"
#include <yaml-cpp/yaml.h>

ScenarioParser::ScenarioParser(const std::string& filepath): m_filepath(filepath) {
}

Config ScenarioParser::parseScenarios(const ReplaceBy& replace_by) {
    std::cout << "Creating root node" << std::endl;
    ConfigNode root = {YAML::LoadFile(m_filepath), true};
    std::cout << "Root node created, calling parseConfig" << std::endl;
    auto result = parseConfig(root, replace_by);
    std::cout << "parseConfig complete" << std::endl;
    return result;
}
