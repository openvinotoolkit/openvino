// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/configuration/parse_numeric.hpp"

#include "ie_plugin_config.hpp"

#include "vpu/utils/error.hpp"

namespace vpu {

int parseInt(const std::string& src) {
    return std::stoi(src);
}

bool Positive(int value) {
    return value > 0;
}

bool Negative(int value) {
    return value < 0;
}

}  // namespace vpu
