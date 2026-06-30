// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

std::string to_lower(std::string value);

bool contains_substring(const std::string& value, const std::string& substring);

std::string format_duration_ms(double value);

std::string partial_shapes_to_string(const std::map<std::string, ov::PartialShape>& shapes);

std::map<std::string, ov::PartialShape> parse_input_shapes(const std::string& shapes_string,
                                                           const std::vector<ov::Output<const ov::Node>>& inputs);