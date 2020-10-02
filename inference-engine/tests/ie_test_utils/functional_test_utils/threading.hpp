// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

void runParallel(std::function<void(void)> func,
                 const unsigned int iterations = 100,
                 const unsigned int threadsNum = 8);
