// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cpu_memory.h"

namespace ov::intel_cpu {
class SubMemoryManager {
public:
    struct MemoryInfo {
        void* send_buf = nullptr;
        bool flag = false;
        bool last_used = false;
    };

    SubMemoryManager(int num_sub_streams) {
        assert(num_sub_streams);
        _num_sub_streams = num_sub_streams;
        MemoryInfo memory_info;
        std::vector<MemoryInfo> memorys;
        memorys.assign(_num_sub_streams, memory_info);
        _memorys_table.assign(2, memorys);
        _use_count.assign(2, 0);
    }

    int get_memory_id(int sub_stream_id) {
        for (int i = 0; i < 2; i++) {
            if (!_memorys_table[i][sub_stream_id].last_used) {
                return i;
            }
        }
        return -1;
    }

    void set_memory_used(int memory_id, int sub_stream_id) {
        _memorys_table[memory_id][sub_stream_id].last_used = true;
        _memorys_table[(memory_id + 1) % 2][sub_stream_id].last_used = false;
    }

    int _num_sub_streams;
    std::vector<std::vector<MemoryInfo>> _memorys_table;
    std::vector<int> _use_count;
    std::mutex _flagMutex;
};
}  // namespace ov::intel_cpu
