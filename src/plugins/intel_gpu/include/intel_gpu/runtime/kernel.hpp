// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_args.hpp"
#include "event.hpp"

#include <memory>
#include <vector>

namespace cldnn {

using kernel_id = std::string;

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const = 0;
    virtual ~kernel() = default;
    virtual std::string get_id() const { return ""; }
};

}  // namespace cldnn
