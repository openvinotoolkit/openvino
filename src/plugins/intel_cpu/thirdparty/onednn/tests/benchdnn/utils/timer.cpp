/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <chrono>

#include "utils/timer.hpp"

namespace timer {

double ms_now() {
    auto timePointTmp
            = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

#if !defined(BENCHDNN_USE_RDPMC) || defined(_WIN32)
unsigned long long ticks_now() {
    return (unsigned long long)0;
}
#else
unsigned long long ticks_now() {
    unsigned eax, edx, ecx;

    ecx = (1 << 30) + 1;
    __asm__ volatile("rdpmc" : "=a"(eax), "=d"(edx) : "c"(ecx));

    return (unsigned long long)eax | (unsigned long long)edx << 32;
}
#endif

void timer_t::reset() {
    times_ = 0;
    for (int i = 0; i < n_modes; ++i)
        ticks_[i] = 0;
    ticks_start_ = 0;
    for (int i = 0; i < n_modes; ++i)
        ms_[i] = 0;
    ms_start_ = 0;

    start();
}

void timer_t::start() {
    ticks_start_ = ticks_now();
    ms_start_ = ms_now();
}

void timer_t::stop(int add_times) {
    if (add_times == 0) return;

    unsigned long long d_ticks = ticks_now() - ticks_start_;
    double d_ms = ms_now() - ms_start_;

    ticks_start_ += d_ticks;
    ms_start_ += d_ms;

    ms_[timer_t::avg] += d_ms;
    ticks_[timer_t::avg] += d_ticks;

    d_ticks /= add_times;
    d_ms /= add_times;

    ms_[timer_t::min] = times_ ? std::min(ms_[timer_t::min], d_ms) : d_ms;
    ms_[timer_t::max] = times_ ? std::max(ms_[timer_t::max], d_ms) : d_ms;

    ticks_[timer_t::min]
            = times_ ? std::min(ticks_[timer_t::min], d_ticks) : d_ticks;
    ticks_[timer_t::max]
            = times_ ? std::max(ticks_[timer_t::max], d_ticks) : d_ticks;

    times_ += add_times;
}

timer_t &timer_t::operator=(const timer_t &rhs) {
    if (this == &rhs) return *this;
    times_ = rhs.times_;
    for (int i = 0; i < n_modes; ++i)
        ticks_[i] = rhs.ticks_[i];
    ticks_start_ = rhs.ticks_start_;
    for (int i = 0; i < n_modes; ++i)
        ms_[i] = rhs.ms_[i];
    ms_start_ = rhs.ms_start_;
    return *this;
}

} // namespace timer
