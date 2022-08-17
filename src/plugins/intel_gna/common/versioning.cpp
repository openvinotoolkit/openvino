// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "versioning.hpp"

#include <sstream>
#include <string>

#include "openvino/core/version.hpp"

namespace GNAPluginNS {
namespace common {

std::string GetVersionOfOv() {
    std::stringstream s;
    s << ov::get_openvino_version();
    return s.str();
}
} // namespace common
} // namespace GNAPluginNS
