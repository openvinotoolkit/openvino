// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/file_util.hpp"
#include <stdexcept>

namespace ov {
namespace intel_gpu {

void save_binary(const std::string &path, std::vector<uint8_t> binary) {
    try {
        ov::util::save_binary(path, binary);
    } catch (std::runtime_error&) {}
}

}  // namespace intel_gpu
}  // namespace ov
