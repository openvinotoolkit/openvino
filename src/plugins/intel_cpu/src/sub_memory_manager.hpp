// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <assert.h>
#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {
class SubMemoryManager {
public:
    struct MemoryInfo {
        void* send_buf;
        std::shared_ptr<void> buf = nullptr;
        bool flag;
        bool last_used;
    };

    SubMemoryManager(int num_sub_streams) {
        assert(num_sub_streams);
        _num_sub_streams = num_sub_streams;
        MemoryInfo memory_info;
        memory_info.flag = false;
        memory_info.last_used = false;
        std::vector<MemoryInfo> memorys;
        memorys.assign(_num_sub_streams, memory_info);
        _memorys_table.assign(2, memorys);
        _use_count.assign(2, 0);
        _shared_memorys.assign(2, {});
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

    void get_buffer(const dnnl::engine& eng, std::shared_ptr<void>& buf, MemoryDescPtr desc) {
        if (buf == nullptr) {
            buf = std::make_shared<Memory>(eng, desc);
        } else {
            MemoryPtr shared_mem = std::static_pointer_cast<Memory>(buf);
            shared_mem->redefineDesc(desc);
        }
    }

    std::shared_ptr<void> get_shared_memory(const dnnl::engine& eng, MemoryDescPtr desc, int sub_stream_id, std::string name) {
        if (_shared_memorys[sub_stream_id].find(name) == _shared_memorys[sub_stream_id].end()) {
            _shared_memorys[sub_stream_id].emplace(std::make_pair(name, nullptr));
        }
        get_buffer(eng, _shared_memorys[sub_stream_id][name], desc);
        return _shared_memorys[sub_stream_id][name];
    }

    std::shared_ptr<void> get_pingpang_memory(const dnnl::engine& eng,
                                              MemoryDescPtr desc,
                                              int switch_id,
                                              int sub_stream_id) {
        get_buffer(eng, _memorys_table[switch_id][sub_stream_id].buf, desc);
        return _memorys_table[switch_id][sub_stream_id].buf;
    }

    int _num_sub_streams;
    std::vector<std::vector<MemoryInfo>> _memorys_table;
    std::vector<std::map<std::string, std::shared_ptr<void>>> _shared_memorys;
    std::vector<int> _use_count;
    std::mutex _flagMutex;
};
}  // namespace intel_cpu

}  // namespace ov