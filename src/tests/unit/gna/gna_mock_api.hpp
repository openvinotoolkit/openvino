// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock-generated-function-mockers.h>

#if defined(_WIN32)
    #ifdef libGNAStubs_EXPORTS
        #define GNA_STUBS_EXPORT __declspec(dllexport)
    #else
        #define GNA_STUBS_EXPORT __declspec(dllimport)
    #endif
#else
    #define GNA_STUBS_EXPORT
#endif

class GNACppApi {
public:
    GNA_STUBS_EXPORT GNACppApi();
    GNA_STUBS_EXPORT ~GNACppApi();

    MOCK_METHOD3(Gna2MemoryAlloc, Gna2Status(
        uint32_t sizeRequested,
        uint32_t * sizeGranted,
        void ** memoryAddress));
    MOCK_METHOD2(Gna2RequestWait, Gna2Status(
        uint32_t requestId,
        uint32_t timeoutMilliseconds));
};
