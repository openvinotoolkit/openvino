//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "parser/parser.hpp"

#include <yaml-cpp/yaml.h>

Config parseConfig(const YAML::Node& root, const ReplaceBy& replace_by);
