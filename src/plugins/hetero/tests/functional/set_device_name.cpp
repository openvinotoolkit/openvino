// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "set_device_name.hpp"

#include <stdexcept>
#include <string>

#include "openvino/core/except.hpp"

namespace ov {
namespace test {
void set_device_suffix(const std::string& suffix) {
    OPENVINO_THROW(suffix.empty(), "The suffix can't be used for TEMPLATE device!");
}
}  // namespace test
}  // namespace ov
