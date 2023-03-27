// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/cpu_map_scheduling.hpp"

#include "ie_system_conf.h"

namespace ov {

std::vector<std::vector<int>> apply_scheduling_core_type(const ov::hint::SchedulingCoreType input_type,
                                                         const std::vector<std::vector<int>> proc_type_table) {
    std::vector<std::vector<int>> result_table = proc_type_table;

    switch (input_type) {
    case ov::hint::SchedulingCoreType::ANY_CORE:
        break;
    case ov::hint::SchedulingCoreType::PCORE_ONLY:
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[EFFICIENT_CORE_PROC];
                i[EFFICIENT_CORE_PROC] = 0;
            }
        }
        break;
    case ov::hint::SchedulingCoreType::ECORE_ONLY:
        if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
            (proc_type_table[0][EFFICIENT_CORE_PROC] != proc_type_table[0][ALL_PROC])) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[MAIN_CORE_PROC] + i[HYPER_THREADING_PROC];
                i[MAIN_CORE_PROC] = 0;
                i[HYPER_THREADING_PROC] = 0;
            }
        }
        break;
    default:
        throw ov::Exception{"Unsupported core type!"};
    }

    return result_table;
}

}  // namespace ov