// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief Thrown in case of cancelled asynchronous operation.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API Cancelled : public Exception {
    using Exception::Exception;
};

/**
 * @brief Thrown in case of calling the InferRequest methods while the request is
 * busy with compute operation.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API Busy : public Exception {
    using Exception::Exception;
};

[[noreturn]] OPENVINO_RUNTIME_API void throw_cancelled(const CheckLocInfo& check_loc_info,
                                                       const std::string& context_info,
                                                       const std::string& explanation);
[[noreturn]] OPENVINO_RUNTIME_API void throw_busy(const CheckLocInfo& check_loc_info,
                                                  const std::string& context_info,
                                                  const std::string& explanation);

#define OPENVINO_CANCELLED(...) OPENVINO_ASSERT_HELPER(::ov::throw_cancelled, "", false, __VA_ARGS__)
#define OPENVINO_BUSY(...)      OPENVINO_ASSERT_HELPER(::ov::throw_busy, "", false, __VA_ARGS__)

}  // namespace ov
