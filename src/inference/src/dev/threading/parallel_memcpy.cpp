// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/parallel_memcpy.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace {
void copy_part(void* destination, const void* source, size_t count, size_t index, size_t parts) {
    size_t begin = 0, end = 0;
    ov::splitter(count, parts, index, begin, end);
    if (begin == end)
        return;

    auto* dst = static_cast<unsigned char*>(destination) + begin;
    const auto* src = static_cast<const unsigned char*>(source) + begin;
#ifdef _WIN32
    memcpy_s(dst, end - begin, src, end - begin);
#else
    std::memcpy(dst, src, end - begin);
#endif
}
}  // namespace

void ov::threading::parallel_memcpy(void* destination, const void* source, size_t count) {
    if (count == 0)
        return;
    ov::parallel_nt(0, [&](size_t index, size_t parts) {
        copy_part(destination, source, count, index, parts);
    });
}

void ov::threading::parallel_memcpy(void* destination,
                                    const void* source,
                                    size_t count,
                                    ITaskExecutor& executor,
                                    size_t num_tasks) {
    if (count == 0)
        return;
    num_tasks = std::min(num_tasks, count);
    if (num_tasks < 2) {
        copy_part(destination, source, count, 0, 1);
        return;
    }

    std::vector<Task> tasks;
    tasks.reserve(num_tasks);
    for (size_t i = 0; i < num_tasks; ++i) {
        tasks.emplace_back([=] {
            copy_part(destination, source, count, i, num_tasks);
        });
    }
    executor.run_and_wait(tasks);
}
