// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
namespace runtime {
/// Thrown in case of canceled asynchronous operation
class OPENVINO_API Cancelled : public Exception {
    using Exception::Exception;
};

/// Thrown in case of busy infer request
class OPENVINO_API Busy : public Exception {
    using Exception::Exception;
};
}  // namespace runtime
}  // namespace ov
