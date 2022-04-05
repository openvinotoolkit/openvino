// Copyright (C) 2018-2022 Intel Corporation
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
    virtual std::shared_ptr<kernel> clone() const = 0;
    virtual ~kernel() = default;
};

}  // namespace cldnn
