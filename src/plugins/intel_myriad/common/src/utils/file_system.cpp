// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/file_system.hpp>

#include <string>

namespace vpu {

std::string fileNameNoExt(const std::string& filePath) {
    auto pos = filePath.rfind('.');
    if (pos == std::string::npos)
        return filePath;
    return filePath.substr(0, pos);
}

}  // namespace vpu
