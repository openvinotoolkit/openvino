// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gna2-api.h>
#include <gmock/gmock.h>
#include <cstdint>

class GNACppApi {
public:
    GNACppApi();
    ~GNACppApi();

    MOCK_METHOD3(Gna2MemoryAlloc, Gna2Status(
        uint32_t sizeRequested,
        uint32_t * sizeGranted,
        void ** memoryAddress));

    MOCK_METHOD1(Gna2MemoryFree, Gna2Status(void* memory));

    MOCK_METHOD1(Gna2DeviceOpen, Gna2Status(
        uint32_t deviceIndex));

    MOCK_METHOD1(Gna2DeviceClose, Gna2Status(uint32_t deviceIndex));

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

    MOCK_METHOD2(Gna2RequestEnqueue, Gna2Status(
        uint32_t requestConfigId,
        uint32_t* requestId));

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
