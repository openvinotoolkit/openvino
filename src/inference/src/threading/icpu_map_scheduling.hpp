// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file icpu_map_scheduling.hpp
 * @brief A header file for internal CPU streams calulation implementation.
 */

#pragma once

#include <vector>

#include "ie_system_conf.h"
#include "threading/ie_cpu_streams_info.hpp"

namespace ov {

/**
 * @brief      Update CPU pinning property according to platform and user setting
 * @param[in]  input_pinning indicates value of property enable_cpu_pinning.
 * @param[in]  input_changed indicates if value is set by user.
 * @param[in]  stream_info_table indicates the stream details that will be created..
 * @param[in]  original_proc_type_table indicates CPU details on this platform
 * @return     updated proc_type_table which removed unmatched processors
 */
bool update_cpu_pinning(const bool input_pinning,
                        const bool input_changed,
                        const std::vector<std::vector<int>>& stream_info_table,
                        const std::vector<std::vector<int>>& original_proc_type_table);

}  // namespace ov