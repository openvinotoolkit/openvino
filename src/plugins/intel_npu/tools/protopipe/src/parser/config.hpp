//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "parser/parser.hpp"
#include "parser/config_node.hpp"

Config parseConfig(const ConfigNode& root, const ReplaceBy& replace_by);
