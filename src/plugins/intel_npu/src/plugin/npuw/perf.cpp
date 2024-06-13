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
    return khr::duration_cast<khr::microseconds>(f - s).count() / 1000.0f;
};

