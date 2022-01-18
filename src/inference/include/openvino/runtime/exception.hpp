// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
namespace runtime {

/**
 * @brief Thrown in case of cancel;ed asynchronous operation
 */
class OPENVINO_RUNTIME_API Cancelled : public Exception {
    using Exception::Exception;
};

/**
 * @brief Thrown in case of calling InferRequest methods while the request is busy with compute operation.
 */
class OPENVINO_RUNTIME_API Busy : public Exception {
    using Exception::Exception;
};
}  // namespace runtime
}  // namespace ov