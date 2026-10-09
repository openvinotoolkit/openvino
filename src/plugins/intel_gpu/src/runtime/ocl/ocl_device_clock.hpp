// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"

#include <chrono>
#include <mutex>
#include <optional>

namespace cldnn {
namespace ocl {

/// @brief Maps the host steady clock onto the OpenCL device profiling clock, so that
/// host-timed synthetic events are comparable with native ones. The correlation is
/// approximated as affine over short refresh intervals and applied arithmetically
/// rather than calling clGetDeviceAndHostTimer() per event, which costs a driver
/// round trip.
class device_clock_sync {
public:
    struct anchor {
        std::chrono::nanoseconds host{0};
        std::chrono::nanoseconds device{0};
        bool valid = false;
    };

    static constexpr std::chrono::nanoseconds default_refresh_interval = std::chrono::milliseconds(100);

    explicit device_clock_sync(const cl::Device& device);

    static std::chrono::nanoseconds host_now() {
        return std::chrono::steady_clock::now().time_since_epoch();
    }

    /// @brief Maps @p host_ts through @p base, refined by @p late when the anchors are
    /// far enough apart and imply a plausible rate. Pure; exposed for testing.
    static std::chrono::nanoseconds interpolate(const anchor& base,
                                                const anchor& late,
                                                std::chrono::nanoseconds host_ts);

    /// @brief False when the device has no usable timer, i.e. no start timestamp can
    /// ever be synthesized.
    bool is_valid() const;

    /// @brief Re-samples the correlation if the last attempt is older than
    /// @p min_interval. The rate limit amortizes the driver query across events.
    void refresh_if_stale(std::chrono::nanoseconds min_interval = default_refresh_interval);

    /// @brief Returns nullopt when the device provides no usable timer.
    std::optional<std::chrono::nanoseconds> to_device(std::chrono::nanoseconds host_ts) const;

private:
    static anchor sample(const cl::Device& device);

    // Immutable after construction, so sample() may read it without the lock.
    cl::Device m_device;

    mutable std::mutex m_mutex;
    anchor m_base;
    anchor m_latest;
    std::chrono::nanoseconds m_last_refresh{0};
};

}  // namespace ocl
}  // namespace cldnn
