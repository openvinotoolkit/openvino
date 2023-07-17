// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define INTEL_GNA_DLLEXPORT 1


#include <gna2-instrumentation-api.h>
#include <gna2-inference-api.h>
#include <gna2-model-export-api.h>

#include "gna_mock_api.hpp"

static GNACppApi * current = nullptr;

GNACppApi :: GNACppApi() {
    current = this;
}

GNACppApi :: ~GNACppApi() {
    current = nullptr;
}

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif

GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t *sizeGranted,
    void **memoryAddress) {
    if (sizeGranted != nullptr) {
        *sizeGranted = sizeRequested;
    }
    if (current != nullptr) {
        return current->Gna2MemoryAlloc(sizeRequested, sizeGranted, memoryAddress);
    }
    *memoryAddress = reinterpret_cast<void*>(1);
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2MemorySetTag(
    void* memory,
    uint32_t tag) {
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceCreateForExport(
    Gna2DeviceVersion targetDeviceVersion,
    uint32_t * deviceIndex) {
    *deviceIndex = 1;
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceOpen(
    uint32_t deviceIndex) {
    if (current != nullptr) {
        return current->Gna2DeviceOpen(deviceIndex);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads) {
    if (current != nullptr) {
        return current->Gna2DeviceSetNumberOfThreads(deviceIndex, numberOfThreads);
    }
    return Gna2StatusSuccess;
}

GNA2_API Gna2Status Gna2DeviceClose(
    uint32_t deviceIndex) {
    if (current != nullptr) {
        return current->Gna2DeviceClose(deviceIndex);
    }
    return Gna2StatusSuccess;
}

GNA2_API Gna2Status Gna2DeviceGetCount(
    uint32_t * numberOfDevices) {
    if (numberOfDevices != nullptr) {
        *numberOfDevices = 1;
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory) {
    if (current != nullptr) {
        return current->Gna2MemoryFree(memory);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2StatusGetMessage(enum Gna2Status status,
    char * messageBuffer, uint32_t messageBufferSize) {
    if (current != nullptr) {
        return current->Gna2StatusGetMessage(status, messageBuffer, messageBufferSize);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelCreate(
    uint32_t deviceIndex,
    struct Gna2Model const * model,
    uint32_t * modelId) {
    if (current != nullptr) {
        return current->Gna2ModelCreate(deviceIndex, model, modelId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelRelease(
    uint32_t modelId) {
    if (current != nullptr) {
        return current->Gna2ModelRelease(modelId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelGetLastError(
    struct Gna2ModelError* error) {
    if (current != nullptr) {
        return current->Gna2ModelGetLastError(error);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigCreate(
    uint32_t modelId,
    uint32_t * requestConfigId) {
    if (current != nullptr) {
        return current->Gna2RequestConfigCreate(modelId, requestConfigId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigEnableActiveList(
    uint32_t requestConfigId,
    uint32_t operationIndex,
    uint32_t numberOfIndices,
    uint32_t const * indices) {
    if (current != nullptr) {
        return current->Gna2RequestConfigEnableActiveList(requestConfigId, operationIndex, numberOfIndices, indices);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestConfigSetAccelerationMode(
    uint32_t requestConfigId,
    enum Gna2AccelerationMode accelerationMode) {
    if (current != nullptr) {
        return current->Gna2RequestConfigSetAccelerationMode(requestConfigId, accelerationMode);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestEnqueue(
    uint32_t requestConfigId,
    uint32_t * requestId) {
    if (current != nullptr) {
        return current->Gna2RequestEnqueue(requestConfigId, requestId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2RequestWait(
    uint32_t requestId,
    uint32_t timeoutMilliseconds) {
    if (current != nullptr) {
        return current->Gna2RequestWait(requestId, timeoutMilliseconds);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * exportConfigId) {
    if (current != nullptr) {
        return current->Gna2ModelExportConfigCreate(userAllocator, exportConfigId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId) {
    if (current != nullptr) {
        return current->Gna2ModelExportConfigRelease(exportConfigId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId) {
    if (current != nullptr) {
        return current->Gna2ModelExportConfigSetSource(exportConfigId, sourceDeviceIndex, sourceModelId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion) {
    if (current != nullptr) {
        return current->Gna2ModelExportConfigSetTarget(exportConfigId, targetDeviceVersion);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize) {
    if (current != nullptr) {
        return current->Gna2ModelExport(exportConfigId, componentType, exportBuffer, exportBufferSize);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2DeviceGetVersion(
    uint32_t deviceIndex,
    enum Gna2DeviceVersion * deviceVersion) {
    if (current != nullptr) {
        return current->Gna2DeviceGetVersion(deviceIndex,deviceVersion);
    }
    *deviceVersion = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    enum Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t * results,
    uint32_t * instrumentationConfigId) {
    if (current != nullptr) {
        return current->Gna2InstrumentationConfigCreate(numberOfInstrumentationPoints, selectedInstrumentationPoints, results, instrumentationConfigId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2InstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId) {
    if (current != nullptr) {
        return current->Gna2InstrumentationConfigAssignToRequestConfig(instrumentationConfigId, requestConfigId);
    }
    return Gna2StatusSuccess;
}

GNA2_API enum Gna2Status Gna2GetLibraryVersion(
    char* versionBuffer,
    uint32_t versionBufferSize) {
    if (current != nullptr) {
        return current->Gna2GetLibraryVersion(versionBuffer, versionBufferSize);
    }
    if (versionBuffer != nullptr && versionBufferSize > 0) {
        versionBuffer[0] = '\0';
        return Gna2StatusSuccess;
    }
    return Gna2StatusNullArgumentNotAllowed;
}

#ifdef __cplusplus
}
#endif
