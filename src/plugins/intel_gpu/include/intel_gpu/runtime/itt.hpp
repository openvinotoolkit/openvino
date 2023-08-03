// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov {
namespace intel_gpu {
namespace itt {

template <openvino::itt::domain_t(*domain)()>
struct ScopedTaskCustomTrack {
    ScopedTaskCustomTrack(openvino::itt::handle_t task, openvino::itt::track_t track, int64_t timestamp_begin, int64_t timestamp_end) noexcept {
        openvino::itt::setTrack(track);
        openvino::itt::taskBeginEx(domain(), task, timestamp_begin);
        openvino::itt::taskEndEx(domain(), timestamp_end);
    }

    ScopedTaskCustomTrack(const std::string& task_name, openvino::itt::track_t track, int64_t timestamp_begin, int64_t timestamp_end) noexcept
        : ScopedTaskCustomTrack(openvino::itt::handle(task_name), track, timestamp_begin, timestamp_end) {}

    ~ScopedTaskCustomTrack() noexcept { openvino::itt::setTrack(nullptr); }

    ScopedTaskCustomTrack(const ScopedTaskCustomTrack&) = delete;
    ScopedTaskCustomTrack& operator=(const ScopedTaskCustomTrack&) = delete;
};
namespace domains {
    OV_ITT_DOMAIN(intel_gpu_plugin);
}  // namespace domains
}  // namespace itt
}  // namespace intel_gpu
}  // namespace ov

#define OV_ITT_SCOPED_TASK_CUSTOM_TRACK(domain, track, handle_or_name, timestamp_begin, timestamp_end)  \
    ov::intel_gpu::itt::ScopedTaskCustomTrack<domain> OV_PP_CAT(itt_scoped_task_custom_track, __LINE__) \
                    (handle_or_name, track, timestamp_begin, timestamp_end)
