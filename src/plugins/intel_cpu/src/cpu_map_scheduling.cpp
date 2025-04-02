// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_map_scheduling.hpp"

#include "cpu_streams_calculation.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

namespace ov::intel_cpu {

std::vector<std::vector<int>> apply_scheduling_core_type(ov::hint::SchedulingCoreType& input_type,
                                                         const std::vector<std::vector<int>>& proc_type_table) {
    std::vector<std::vector<int>> result_table = proc_type_table;

    auto update_proc_type_table = [&]() {
        switch (input_type) {
        case ov::hint::SchedulingCoreType::PCORE_ONLY:
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[EFFICIENT_CORE_PROC];
                i[EFFICIENT_CORE_PROC] = 0;
            }
            break;
        case ov::hint::SchedulingCoreType::ECORE_ONLY:
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[MAIN_CORE_PROC] + i[HYPER_THREADING_PROC];
                i[MAIN_CORE_PROC] = 0;
                i[HYPER_THREADING_PROC] = 0;
            }
            break;
        default:
            break;
        }
    };

    if (((input_type == ov::hint::SchedulingCoreType::PCORE_ONLY) && (proc_type_table[0][MAIN_CORE_PROC] == 0)) ||
        ((input_type == ov::hint::SchedulingCoreType::ECORE_ONLY) && (proc_type_table[0][EFFICIENT_CORE_PROC] == 0))) {
        input_type = ov::hint::SchedulingCoreType::ANY_CORE;
    }

    update_proc_type_table();

    return result_table;
}

std::vector<std::vector<int>> apply_hyper_threading(bool& input_ht_hint,
                                                    const bool input_ht_changed,
                                                    const std::string& input_pm_hint,
                                                    const std::vector<std::vector<int>>& proc_type_table) {
    std::vector<std::vector<int>> result_table = proc_type_table;

    if (proc_type_table[0][HYPER_THREADING_PROC] > 0) {
        if (((!input_ht_hint) && input_ht_changed) || ((!input_ht_changed) && (input_pm_hint == "LATENCY")) ||
            ((!input_ht_changed) && (input_pm_hint == "THROUGHPUT") && (proc_type_table.size() > 1))) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[HYPER_THREADING_PROC];
                i[HYPER_THREADING_PROC] = 0;
            }
            input_ht_hint = false;
        } else {
            input_ht_hint = true;
        }
    } else {
        input_ht_hint = false;
    }

    return result_table;
}

bool check_cpu_pinning(const bool cpu_pinning,
                       const bool cpu_pinning_changed,
                       const bool cpu_reservation,
                       const std::vector<std::vector<int>>& streams_info_table) {
    bool result_value;

#if defined(__APPLE__)
    result_value = false;
#elif defined(_WIN32)
    result_value = cpu_pinning_changed ? cpu_pinning : cpu_reservation;
#else
    // The following code disables pinning in case stream contains both Pcore and Ecore
    auto hyper_cores_in_stream = [&]() {
        if (streams_info_table.size() >= 3) {
            if ((streams_info_table[0][PROC_TYPE] == ALL_PROC) &&
                (streams_info_table[1][PROC_TYPE] != EFFICIENT_CORE_PROC) &&
                (streams_info_table[2][PROC_TYPE] == EFFICIENT_CORE_PROC)) {
                return true;
            }
        }
        return false;
    };

    result_value = cpu_pinning_changed ? cpu_pinning : true;
    if (!cpu_pinning_changed && !cpu_reservation) {
        if (hyper_cores_in_stream()) {
            result_value = false;
        }
    }
#endif

    return result_value;
}

}  // namespace ov::intel_cpu
