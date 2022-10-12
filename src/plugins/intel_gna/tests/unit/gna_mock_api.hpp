// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock-generated-function-mockers.h>
#include <gna2-instrumentation-api.h>
#include <gna2-model-export-api.h>

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

    MOCK_METHOD1(Gna2DeviceOpen, Gna2Status(
        uint32_t deviceIndex));

    MOCK_METHOD3(Gna2ModelCreate, Gna2Status(
        uint32_t deviceIndex,
        struct Gna2Model const * model,
        uint32_t * modelId));

    MOCK_METHOD2(Gna2RequestConfigCreate, Gna2Status(
        uint32_t modelId,
        uint32_t * requestConfigId));

    MOCK_METHOD2(Gna2RequestWait, Gna2Status(
        uint32_t requestId,
        uint32_t timeoutMilliseconds));

    MOCK_METHOD2(Gna2DeviceGetVersion, Gna2Status(
        uint32_t deviceIndex,
        enum Gna2DeviceVersion * deviceVersion));

    MOCK_METHOD4(Gna2InstrumentationConfigCreate, Gna2Status(
        uint32_t numberOfInstrumentationPoints,
        enum Gna2InstrumentationPoint* selectedInstrumentationPoints,
        uint64_t * results,
        uint32_t * instrumentationConfigId));

    MOCK_METHOD2(Gna2InstrumentationConfigAssignToRequestConfig, Gna2Status(
        uint32_t instrumentationConfigId,
        uint32_t requestConfigId));

    MOCK_METHOD2(Gna2GetLibraryVersion, Gna2Status(
        char* versionBuffer,
        uint32_t versionBufferSize));
};
