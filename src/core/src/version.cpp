// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/version.hpp"

namespace ov {
const char* OV_VERSION_NUMBER = CI_BUILD_NUMBER;

std::ostream& operator<<(std::ostream& s, const Version& version) {
    s << version.description << std::endl;
    s << "    Version : ";
    s << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH;
    s << std::endl;
    s << "    Build   : ";
    s << version.buildNumber << std::endl;
    return s;
}

std::ostream& operator<<(std::ostream& s, const std::map<std::string, Version>& versions) {
    for (auto&& version : versions) {
        s << version.second << std::endl;
    }
    return s;
}

const Version get_openvino_version() noexcept {
    static const Version version = {OV_VERSION_NUMBER, "OpenVINO Runtime"};
    return version;
}

}  // namespace ov
