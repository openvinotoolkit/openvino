// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "icpu_map_scheduling.hpp"

namespace ov {

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