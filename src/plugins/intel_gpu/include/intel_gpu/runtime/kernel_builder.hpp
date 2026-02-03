// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

/// @brief Defines possible kernel formats
enum class KernelFormat {
    SOURCE,     ///< source code format
    NATIVE_BIN, ///< device native binary format
};

/// @brief Interface for building the GPU kernels. Implementations must be thread-safe to support case where multiple threads use single builder.
class kernel_builder {
public:
    virtual void build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const = 0;
};

}  // namespace cldnn
