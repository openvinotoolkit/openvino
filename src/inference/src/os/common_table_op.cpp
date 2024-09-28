// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_table_op.hpp"

#include "openvino/runtime/system_conf.hpp"

namespace ov {

void update_table_for_proc(const int _processor_id,
                           std::vector<std::vector<int>>& _proc_type_table,
                           const std::vector<std::vector<int>>& _cpu_mapping_table) {
    int current_numa_node = 0;
    int current_socket = 0;

    for (auto& row : _cpu_mapping_table) {
        if (_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            current_numa_node = row[CPU_MAP_NUMA_NODE_ID];
            current_socket = row[CPU_MAP_SOCKET_ID];
            break;
        }
    }
    for (size_t i = 1; i < _proc_type_table.size(); i++) {
        if ((current_numa_node == _proc_type_table[i][PROC_NUMA_NODE_ID]) &&
            (current_socket == _proc_type_table[i][PROC_SOCKET_ID])) {
            std::rotate(_proc_type_table.begin() + 1, _proc_type_table.begin() + i, _proc_type_table.end());
            break;
        }
    }
};

}  // namespace ov
