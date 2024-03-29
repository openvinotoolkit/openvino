// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>

#include <string>

namespace intel_npu {

const std::string ze_result_to_string(const ze_result_t result);

const std::string ze_result_to_description(const ze_result_t result);

}  // namespace intel_npu
