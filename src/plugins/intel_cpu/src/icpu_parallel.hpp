// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

namespace ov::intel_cpu {

class ICpuParallel {
public:
    virtual ~ICpuParallel() = default;

    virtual int get_num_threads() const = 0;
    virtual void parallel_simple(int n, const std::function<void(int, int)>& fn) const = 0;
};

} // namespace ov::intel_cpu
