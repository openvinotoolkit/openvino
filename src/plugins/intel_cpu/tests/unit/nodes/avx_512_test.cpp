// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <iostream>
#include <vector>

#include "nodes/common/cpu_convert.h"
#include "gtest/gtest.h"

using namespace std::chrono;
using namespace ov::intel_cpu;

TEST(cpu_convert, AVX512_fp16_load_store) {
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    for (size_t size : sizes) {
        std::vector<float> input_data(size, 1.23f);
        std::vector<short> output_data(size);

        auto start = high_resolution_clock::now();
        cpu_convert(input_data.data(), output_data.data(), ov::element::f32, ov::element::f16, size);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "size " << size << ": " << duration.count() << " microseconds\n";
    }
}
