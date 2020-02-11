// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <details/ie_exception.hpp>

#include <vpu/utils/io.hpp>

namespace vpu {

// TODO: replace with VPU_THROW_FORMAT/VPU_THROW_UNLESS/VPU_INTERNAL_CHECK and remove
#define VPU_THROW_EXCEPTION THROW_IE_EXCEPTION

namespace details {

template <typename... Args>
[[noreturn]]
void throwFormat(
        const char* filename, int line,
        const char* msg_format,
        const Args&... args) {
    throw InferenceEngine::details::InferenceEngineException(filename, line)
        << formatString(msg_format, args...);
}

}  // namespace details

#define VPU_THROW_FORMAT(...) \
    vpu::details::throwFormat(__FILE__, __LINE__, __VA_ARGS__)

namespace details {

template <typename... Args>
void throwUnless(
        const char* filename, int line,
        bool cond, const char* cond_str,
        const char* msg_prefix,
        const char* msg_format,
        const Args&... args) {
    if (!cond) {
        throw InferenceEngine::details::InferenceEngineException(filename, line)
            << msg_prefix
            << "Check (" << cond_str << ") failed: "
            << formatString(msg_format, args...);
    }
}

}  // namespace details

#define VPU_THROW_UNLESS(cond, ...) \
    vpu::details::throwUnless(__FILE__, __LINE__, cond, #cond, "", __VA_ARGS__)

#ifdef NDEBUG
#   define VPU_INTERNAL_CHECK(cond, ...) \
        vpu::details::throwUnless(__FILE__, __LINE__, cond, #cond, "[Internal Error] ", __VA_ARGS__)
#else
#   define VPU_INTERNAL_CHECK(cond, ...) \
        assert(cond)
#endif

}  // namespace vpu
