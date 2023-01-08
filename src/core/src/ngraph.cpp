// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "ngraph/util.hpp"
#include "ngraph/version.hpp"
#include "openvino/core/version.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

const char* get_ngraph_version_string() {
    return ov::get_openvino_version().buildNumber;
}

void ngraph::get_version(size_t& major, size_t& minor, size_t& patch, std::string& extra) {
    std::string version = get_ngraph_version_string();
    ngraph::parse_version_string(version, major, minor, patch, extra);
}

NGRAPH_SUPPRESS_DEPRECATED_END
