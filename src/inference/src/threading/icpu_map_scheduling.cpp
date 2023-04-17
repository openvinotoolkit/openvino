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

    if (input_changed && !input_pinning) {
        return false;
    } else {
        for (auto& row : stream_info_table) {
            if (row[InferenceEngine::PROC_TYPE] == ALL_PROC) {
                return false;
            }
        }
        return true;
    }

#endif
}

}  // namespace ov