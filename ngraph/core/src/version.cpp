// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/version.hpp"

#include "ngraph/version.hpp"

const char* NGRAPH_VERSION_NUMBER = CI_BUILD_NUMBER;

namespace ov {

const Version* get_openvino_version() noexcept {
    static const Version version = { NGRAPH_VERSION_NUMBER, "OpenVINO Runtime" };
    return &version;
}

}  // namespace ov
