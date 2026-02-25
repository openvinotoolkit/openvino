// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/util/file_util.hpp"

namespace ov::intel_gpu {

// Version of save_binary that don't trow an exception if attempt to open file fails
void save_binary(const std::string& path, const std::vector<uint8_t>& binary);

}  // namespace ov::intel_gpu
