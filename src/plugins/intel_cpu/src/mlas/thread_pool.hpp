// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "mlas.h"

namespace ov {
namespace cpu {
class OVMlasThreadPool : public IMlasThreadPool {
public:
    OVMlasThreadPool() = delete;
    explicit OVMlasThreadPool(const size_t& threadNum) : threadNum(threadNum) {}
    size_t DegreeOfParallelism() override;
    void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) override;

public:
    // the actual threads used for sgemm
    size_t threadNum = 0;
};
};  // namespace cpu
};  // namespace ov