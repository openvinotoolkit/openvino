// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

namespace cldnn {

// Whether the compiler accepts -ze-exp-register-file-size (Xe3+); probed once per device
bool query_register_file_size_option_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);

}  // namespace cldnn
