// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "set_device_name.hpp"

#include <stdexcept>
#include <string>

namespace ov {
namespace test {
void set_device_suffix(const std::string& suffix) {
    if (!suffix.empty()) {
        throw std::runtime_error("The suffix can't be used for CPU device!");
    }
}
}  // namespace test
}  // namespace ov
