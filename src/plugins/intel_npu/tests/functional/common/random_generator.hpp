// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>

class RandomGenerator {
public:
    static std::mt19937& get();

private:
    static RandomGenerator& getInstance();

    RandomGenerator() = default;
    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;
    ~RandomGenerator() = default;

    std::mt19937 generator{std::random_device{}()};
};
