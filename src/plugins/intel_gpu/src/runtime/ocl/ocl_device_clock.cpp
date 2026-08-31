// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_device_clock.hpp"

#include "CL/cl.h"

#include <cstdint>

using namespace cldnn::ocl;

device_clock_sync::device_clock_sync(const cl::Device& device) : m_device(device) {
    m_base = sample(m_device);
    m_last_refresh = host_now();
}

// Cristian's algorithm: attribute the device reading to the midpoint of the host
// interval bracketing the call, keeping the attempt with the tightest bracket.
device_clock_sync::anchor device_clock_sync::sample(const cl::Device& device) {
    constexpr int sample_attempts = 5;

    anchor best;
#if defined(CL_VERSION_2_1)
    if (device.get() == nullptr)
        return best;

    auto best_rtt = std::chrono::nanoseconds::max();
    for (int i = 0; i < sample_attempts; ++i) {
        const auto h0 = host_now();
        cl_ulong device_ts = 0;
        cl_ulong host_ts = 0;  // unused: clGetHostTimer domain, unrelated to steady_clock
        // Devices may lack the OpenCL 2.1 timer even when the headers have it.
        if (clGetDeviceAndHostTimer(device.get(), &device_ts, &host_ts) != CL_SUCCESS)
            break;
        const auto h1 = host_now();

        const auto rtt = h1 - h0;
        if (rtt < best_rtt) {
            best_rtt = rtt;
            best.host = h0 + rtt / 2;
            best.device = std::chrono::nanoseconds(static_cast<int64_t>(device_ts));
            best.valid = true;
        }
    }
#else
    (void)device;
#endif
    return best;
}

bool device_clock_sync::is_valid() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_base.valid;
}

void device_clock_sync::refresh_if_stale(std::chrono::nanoseconds min_interval) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_base.valid)
            return;
        const auto now = host_now();
        if (now - m_last_refresh < min_interval)
            return;
        // Claim the refresh window before releasing the lock so concurrent callers
        // do not stack up driver calls. A failed or superseded sample still consumes
        // this window intentionally; retrying for every event would restore the
        // profiling regression this cache avoids.
        m_last_refresh = now;
    }

    const auto fresh = sample(m_device);

    std::lock_guard<std::mutex> lock(m_mutex);
    // Concurrent samplers may finish out of order; never step back to an older anchor.
    if (fresh.valid && fresh.host > m_base.host && fresh.host > m_latest.host)
        m_latest = fresh;
}

std::chrono::nanoseconds device_clock_sync::interpolate(const anchor& base,
                                                        const anchor& late,
                                                        std::chrono::nanoseconds host_ts) {
    // Closer than this the two anchors carry no rate signal, so only the offset is used.
    constexpr std::chrono::nanoseconds min_rate_span = std::chrono::milliseconds(1);
    // Real drift is a few ppm; anything outside this range indicates a bogus sample.
    constexpr double min_plausible_rate = 0.9;
    constexpr double max_plausible_rate = 1.1;

    auto delta = host_ts - base.host;

    // A second anchor gives the rate as well as the offset, cancelling drift.
    if (late.valid) {
        const auto span = late.host - base.host;
        if (span > min_rate_span) {
            const double rate = static_cast<double>((late.device - base.device).count()) /
                                static_cast<double>(span.count());
            if (rate > min_plausible_rate && rate < max_plausible_rate) {
                delta = std::chrono::nanoseconds(
                    static_cast<int64_t>(static_cast<double>(delta.count()) * rate));
            }
        }
    }

    return base.device + delta;
}

std::optional<std::chrono::nanoseconds> device_clock_sync::to_device(std::chrono::nanoseconds host_ts) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_base.valid)
        return std::nullopt;

    return interpolate(m_base, m_latest, host_ts);
}
