// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/file_util.hpp"
#include <stdexcept>

namespace ov::intel_gpu {

void save_binary(const std::string &path, const std::vector<uint8_t>& binary) {
    try {
        ov::util::save_binary(ov::util::make_path(path), binary.data(), binary.size());
    } catch (std::runtime_error&) {}
}

}  // namespace ov::intel_gpu
