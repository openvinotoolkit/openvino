// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading.hpp"

#include <thread>
#include <vector>

void runParallel(std::function<void(void)> func,
                 const unsigned int iterations,
                 const unsigned int threadsNum) {
    std::vector<std::thread> threads(threadsNum);

    for (auto & thread : threads) {
        thread = std::thread([&](){
            for (unsigned int i = 0; i < iterations; ++i) {
                func();
            }
        });
    }

    for (auto & thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}
