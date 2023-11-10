// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_map_scheduling.hpp"

#include "cpu_streams_calculation.hpp"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace intel_cpu {

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
                                                    const std::string input_pm_hint,
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

bool get_cpu_pinning(bool& input_value,
                     const bool input_changed,
                     const int num_streams,
                     const Config::LatencyThreadingMode latency_threading_mode,
                     const std::vector<std::vector<int>>& proc_type_table) {
    int result_value;
    int num_sockets = get_default_latency_streams(latency_threading_mode);
    bool latency = num_streams <= num_sockets && num_streams > 0;

    if (input_changed) {
        result_value = input_value;
    } else {
        result_value = true;
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 &&
            proc_type_table[0][EFFICIENT_CORE_PROC] < proc_type_table[0][ALL_PROC]) {
            result_value = latency ? false : true;
        }
    }
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
#    if defined(_WIN32)
    if (proc_type_table.size() > 1) {
        result_value = false;
    }
#    endif
#    if defined(__APPLE__)
    result_value = false;
#    endif
#endif
    input_value = result_value;

    return result_value;
}

}  // namespace intel_cpu
}  // namespace ov
