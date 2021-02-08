// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/shape_io.hpp"

namespace vpu {

std::string createIOShapeName(std::string srcName) {
    return srcName + "@shape";
}

bool isIOShapeName(std::string name) {
    return name.find("@shape") != std::string::npos;
}

} // namespace vpu
