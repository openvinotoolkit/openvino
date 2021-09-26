// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "ngraph/util.hpp"
#include "ngraph/version.hpp"
#include "openvino/core/version.hpp"

extern "C" NGRAPH_API const char* get_ngraph_version_string() {
    return ov::get_openvino_version()->buildNumber;
}

namespace ngraph {
NGRAPH_API void get_version(size_t& major, size_t& minor, size_t& patch, std::string& extra) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    std::string version = get_ngraph_version_string();
    ngraph::parse_version_string(version, major, minor, patch, extra);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace ngraph
