// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ze_api.h"

namespace vpux {

const std::string ze_result_to_string(const ze_result_t result);

const std::string ze_result_to_description(const ze_result_t result);

}  // namespace vpux
