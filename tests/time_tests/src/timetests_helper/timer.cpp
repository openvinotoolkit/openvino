// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "timetests_helper/timer.h"
#include <chrono>
#include <fstream>
#include <memory>
#include <string>

#include "statistics_writer.h"

using time_point = std::chrono::high_resolution_clock::time_point;

namespace TimeTest {

Timer::Timer(const std::string &timer_name) {
    name = timer_name;
    start_time = std::chrono::high_resolution_clock::now();
    StatisticsWriter::Instance().addTimer(name);
}

Timer::~Timer() {
    float duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time)
            .count();
    StatisticsWriter::Instance().deleteTimer({name, duration});
}

} // namespace TimeTest
