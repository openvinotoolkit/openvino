// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>

#include "ie_system_conf.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> apply_processor_type(const ProcessorType input_type,
                                                   const std::vector<std::vector<int>> proc_type_table) {
    std::vector<std::vector<int>> result_table = proc_type_table;

    switch (input_type) {
    case ProcessorType::UNDEFINED:
        if ((proc_type_table.size() > 1) && (proc_type_table[0][HYPER_THREADING_PROC] > 0)) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[HYPER_THREADING_PROC];
                i[HYPER_THREADING_PROC] = 0;
            }
        }
        break;
    case ProcessorType::ALL_CORE:
        break;
    case ProcessorType::PHY_CORE_ONLY:
        if (proc_type_table[0][HYPER_THREADING_PROC] > 0) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[HYPER_THREADING_PROC];
                i[HYPER_THREADING_PROC] = 0;
            }
        }
        break;
    case ProcessorType::P_CORE_ONLY:
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[EFFICIENT_CORE_PROC];
                i[EFFICIENT_CORE_PROC] = 0;
            }
        }
        break;
    case ProcessorType::E_CORE_ONLY:
        if ((proc_type_table[0][MAIN_CORE_PROC] > 0) || (proc_type_table[0][HYPER_THREADING_PROC] > 0)) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[MAIN_CORE_PROC] + i[HYPER_THREADING_PROC];
                i[MAIN_CORE_PROC] = 0;
                i[HYPER_THREADING_PROC] = 0;
            }
        }
        break;
    case ProcessorType::PHY_P_CORE_ONLY:
        if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) || (proc_type_table[0][HYPER_THREADING_PROC] > 0)) {
            for (auto& i : result_table) {
                i[ALL_PROC] -= i[EFFICIENT_CORE_PROC] + i[HYPER_THREADING_PROC];
                i[EFFICIENT_CORE_PROC] = 0;
                i[HYPER_THREADING_PROC] = 0;
            }
        }
        break;
    }

    return result_table;
}
}  // namespace intel_cpu
}  // namespace ov