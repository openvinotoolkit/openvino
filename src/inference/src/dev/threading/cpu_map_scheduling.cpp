// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/cpu_map_scheduling.hpp"

#include "threading/ie_cpu_streams_info.hpp"
#include "ie_system_conf.h"

namespace ov {

std::vector<std::vector<int>> apply_scheduling_core_type(const ov::hint::SchedulingCoreType input_type,
                                                         const std::vector<std::vector<int>>& proc_type_table) {
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

std::vector<std::vector<int>> apply_hyper_threading(bool input_value,
                                                    const bool input_changed,
                                                    const std::vector<std::vector<int>>& proc_type_table) {
    std::vector<std::vector<int>> result_table = proc_type_table;

    if ((proc_type_table[0][HYPER_THREADING_PROC] > 0) &&
        (((!input_value) && input_changed) || ((!input_changed) && (proc_type_table.size() > 1)))) {
        for (auto& i : result_table) {
            i[ALL_PROC] -= i[HYPER_THREADING_PROC];
            i[HYPER_THREADING_PROC] = 0;
        }
        input_value = false;
    }

    return result_table;
}

bool update_cpu_pinning(const bool input_pinning,
                        const bool input_changed,
                        const std::vector<std::vector<int>>& stream_info_table,
                        const std::vector<std::vector<int>>& original_proc_type_table) {
#if (defined(_WIN32) || defined(_WIN64))

    return false;

#endif

#ifdef __linux__

    if (original_proc_type_table[0][HYPER_THREADING_PROC] == 0) {
        return false;
    } else if (!input_changed) {
        for (auto& row : stream_info_table) {
            if ((row[InferenceEngine::PROC_TYPE] == ALL_PROC) || (row[InferenceEngine::PROC_TYPE] == MAIN_CORE_PROC) ||
                (row[InferenceEngine::PROC_TYPE] == HYPER_THREADING_PROC)) {
                return false;
            }
        }
        return true;
    } else {
        return input_pinning;
    }

#endif
}

}  // namespace ov