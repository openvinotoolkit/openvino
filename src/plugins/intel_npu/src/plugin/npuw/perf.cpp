// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>

#include "perf.hpp"

float ov::npuw::perf::ms_to_run(std::function<void()> &&body) {
    namespace khr = std::chrono;
    const auto s = khr::steady_clock::now();
    body();
    const auto f = khr::steady_clock::now();
    const std::chrono::duration<double> diff = f - s;
    return khr::duration_cast<khr::microseconds>(diff).count() / 1000.0;
};

