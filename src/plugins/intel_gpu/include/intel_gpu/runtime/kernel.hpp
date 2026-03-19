// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>

namespace cldnn {

using kernel_id = std::string;

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const = 0;
    /// @brief Check if objects share the same handle to the kernel instance
    /// @param other kernel object for comparison
    /// @return true if underlying kernel handles are the same, false otherwise
    virtual bool is_same(const kernel &other) const = 0;
    virtual ~kernel() = default;

    virtual std::string get_id() const = 0;
    virtual std::vector<uint8_t> get_binary() const = 0;
    virtual std::string get_build_log() const = 0;
};

}  // namespace cldnn

namespace ov::intel_gpu {
using Kernel = cldnn::kernel;
}
