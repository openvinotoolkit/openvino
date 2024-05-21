// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_generator.hpp"

RandomGenerator& RandomGenerator::getInstance() {
    static RandomGenerator instance;
    return instance;
}

std::mt19937& RandomGenerator::get() {
    return getInstance().generator;
}
