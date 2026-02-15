//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "parser/parser.hpp"

#include <yaml-cpp/yaml.h>

Config parseConfig(const YAML::Node& root, const ReplaceBy& replace_by);
