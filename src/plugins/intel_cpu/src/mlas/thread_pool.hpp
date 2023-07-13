// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ov {
namespace cpu {
class ThreadPool {
public:
    ThreadPool() = delete;
    explicit ThreadPool(const size_t& threadNum) : threadNum(threadNum) {}
public:
    // the actual threads used for sgemm
    size_t threadNum = 0;
};
size_t DegreeOfParallelism(ThreadPool* tp);
void TrySimpleParallelFor(ThreadPool* tp, const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);
size_t getCacheSize(int level, bool perCore);
};  // namespace cpu
};  // namespace ov