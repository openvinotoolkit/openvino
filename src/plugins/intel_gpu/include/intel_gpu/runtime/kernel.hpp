// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

namespace cldnn {

using kernel_id = std::string;

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const = 0;
    virtual ~kernel() = default;

    virtual std::string get_id() const = 0;
    virtual std::vector<uint8_t> get_binary() const = 0;
};

}  // namespace cldnn
