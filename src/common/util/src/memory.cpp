// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/memory.hpp"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "memory_prefetch.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::util {
void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept;

void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    const auto chunk_size = std::max<size_t>(align_size_up(size / num_threads, page_size), one_mib);

    std::vector<std::thread> threads;
    threads.reserve(ceil_div(size, chunk_size));

    for (auto first = reinterpret_cast<const uint8_t*>(ptr), last = first + size; first < last; first += chunk_size) {
        threads.emplace_back(PageToucher{first, std::min(first + chunk_size, last), page_size});
    }
    for (auto& t : threads) {
        t.join();
    }
}

}  // namespace ov::util
