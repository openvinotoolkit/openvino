// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/util/file_util.hpp"

namespace ov {
ov::util::FilePath getFilePathFromLibDir(const std::string& filePath);
ov::util::FilePath getPluginPath(const std::string& pluginName);
ov::util::FilePath getPluginPathFromXML(const std::string& pluginPath);
}  // namespace ov