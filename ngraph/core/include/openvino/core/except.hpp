// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <stdexcept>

#include "openvino/core/core_visibility.hpp"

namespace ov {
/// Base error for ov runtime errors.
class OPENVINO_API Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& what_arg) : std::runtime_error(what_arg) {}

    explicit Exception(const char* what_arg) : std::runtime_error(what_arg) {}

    explicit Exception(const std::stringstream& what_arg) : std::runtime_error(what_arg.str()) {}
};

}  // namespace ov
