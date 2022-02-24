// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>


#include <vpu/utils/io.hpp>

#include <string>
#include <memory>
#include <utility>

#include <ie_common.h>

namespace vpu {

// TODO: replace with VPU_THROW_FORMAT/VPU_THROW_UNLESS/VPU_INTERNAL_CHECK and remove
#define VPU_THROW_EXCEPTION IE_THROW()

namespace details {

using VPUException = InferenceEngine::Exception;

class UnsupportedLayerException : public VPUException {
public:
    using VPUException::VPUException;
};

class UnsupportedConfigurationOptionException : public VPUException {
public:
    using VPUException::VPUException;
};

template <class Exception, typename... Args>
void throwFormat(const char* fileName, int lineNumber, const char* messageFormat, Args&&... args) {
    IE_THROW(GeneralError) << '\n' << fileName  << ':' << lineNumber << ' '
                      << formatString(messageFormat, std::forward<Args>(args)...);
}

}  // namespace details

#define VPU_THROW_FORMAT(...)                                                                   \
    ::vpu::details::throwFormat<::vpu::details::VPUException>(__FILE__, __LINE__, __VA_ARGS__)

#define VPU_THROW_UNLESS(condition, ...)                                                                \
    do {                                                                                                \
        if (!(condition)) {                                                                             \
            ::vpu::details::throwFormat<::vpu::details::VPUException>(__FILE__, __LINE__, __VA_ARGS__); \
        }                                                                                               \
    } while (false)

#define VPU_THROW_UNSUPPORTED_LAYER_UNLESS(condition, ...)                                                              \
    do {                                                                                                                \
        if (!(condition)) {                                                                                             \
            ::vpu::details::throwFormat<::vpu::details::UnsupportedLayerException>(__FILE__, __LINE__, __VA_ARGS__);    \
        }                                                                                                               \
    } while (false)

#define VPU_THROW_UNSUPPORTED_OPTION_UNLESS(condition, ...)                                                                           \
    do {                                                                                                                              \
        if (!(condition)) {                                                                                                           \
            ::vpu::details::throwFormat<::vpu::details::UnsupportedConfigurationOptionException>(__FILE__, __LINE__, __VA_ARGS__);    \
        }                                                                                                                             \
    } while (false)

#ifdef NDEBUG
#   define VPU_INTERNAL_CHECK(condition, ...)                               \
        do {                                                                \
            if (!(condition)) {                                             \
                ::vpu::details::throwFormat<::vpu::details::VPUException>(  \
                    __FILE__, __LINE__,                                     \
                    "[Internal Error]: " __VA_ARGS__);                      \
            }                                                               \
        } while (false)
#else
#   define VPU_INTERNAL_CHECK(condition, ...)                       \
        assert((condition) || !::vpu::formatString(__VA_ARGS__).empty())
#endif

#ifdef NDEBUG
#   define VPU_INTERNAL_FAIL(...)                                   \
        ::vpu::details::throwFormat<::vpu::details::VPUException>(  \
            __FILE__, __LINE__,                                     \
            "[Internal Error] Unreachable code: " __VA_ARGS__)
#else
#   define VPU_INTERNAL_FAIL(...)                           \
        assert(false && !::vpu::formatString(__VA_ARGS__).empty())
#endif

}  // namespace vpu
