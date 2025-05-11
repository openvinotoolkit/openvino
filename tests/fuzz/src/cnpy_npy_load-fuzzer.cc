// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <cnpy.h>
#include "fuzz-utils.h"


extern "C" int LLVMFuzzerTestOneInput(const uint8_t * inputData, size_t inputSize) {
    MemoryFile file(inputData, inputSize);
    
    try {
        cnpy::NpyArray array = cnpy::npy_load(file.name());
    }
    catch (const std::exception&) {
        return 0;  // fail gracefully on expected exceptions
    }

    return 0;
}
