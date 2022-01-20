// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/utils/enums.hpp"

namespace vpu {

VPU_DECLARE_ENUM(LogLevel,
    None,
    Fatal,    /* used for very severe error events that will most probably cause the application to terminate */
    Error,    /* reporting events which are not expected during normal execution, containing probable reason */
    Warning,  /* indicating events which are not usual and might lead to errors later */
    Info,     /* short enough messages about ongoing activity in the process */
    Debug,    /* more fine-grained messages with references to particular data and explanations */
    Trace     /* involved and detailed information about execution, helps to trace the execution flow, produces huge output */
)

}  // namespace vpu
