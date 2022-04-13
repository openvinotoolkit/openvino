// Copyright (C) 2018-2022 Intel Corporation
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

}  // namespace ov
