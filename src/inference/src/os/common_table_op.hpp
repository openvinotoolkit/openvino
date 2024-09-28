// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides a set of operation for cpu_mapping_table and proc_type_table.
 * @file common_table_op.hpp
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

namespace ov {

/**
 * @brief     Move processors type information for current numa node and socket to the top of the table
 * @param[in] _processor_id current processor id
 * @param[in] _proc_type_table summary table of number of processors per type
 * @param[in] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void update_table_for_proc(const int _processor_id,
                           std::vector<std::vector<int>>& _proc_type_table,
                           const std::vector<std::vector<int>>& _cpu_mapping_table);

}  // namespace ov