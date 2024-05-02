// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "event.hpp"
#include "kernel_args.hpp"

namespace cldnn {

using kernel_id = std::string;

class kernel {
public:
    using ptr = std::shared_ptr<kernel>;
    virtual std::shared_ptr<kernel> clone() const = 0;
    virtual ~kernel() = default;
    virtual std::string get_id() const {
        return "";
    }
};

}  // namespace cldnn
