// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <thread>
#include <list>

#include <ie_common.h>

#include "memory/gna_mem_requests.hpp"

#include "gna2-common-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"

#include "gna2-memory-api.h"
#include "gna2-model-api.h"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"

enum GnaWaitStatus : int {
    GNA_REQUEST_COMPLETED = 0,  // and removed from GNA library queue
    GNA_REQUEST_ABORTED = 1,    // for QoS purposes
    GNA_REQUEST_PENDING = 2     // for device busy purposes
};

struct GnaAllocation {
    void* ptr = nullptr;
    size_t sizeRequested = 0;
    size_t sizeGranted = 0;
    void SetTag(Gna2MemoryTag in) {
        isTagSet = true;
        tag = in;
    }
    bool isTag(Gna2MemoryTag in) {
        return isTagSet && in == tag;
    }
    std::string GetTagName() const {
        static const std::map< Gna2MemoryTag, std::string > tm = {
                { Gna2MemoryTagReadWrite, "Gna2MemoryTagReadWrite" },
                { Gna2MemoryTagInput, "Gna2MemoryTagInput" },
                { Gna2MemoryTagOutput, "Gna2MemoryTagOutput" },
                { Gna2MemoryTagReadOnly, "Gna2MemoryTagReadOnly" },
                { Gna2MemoryTagExternalBufferInput, "Gna2MemoryTagExternalBufferInput" },
                { Gna2MemoryTagExternalBufferOutput, "Gna2MemoryTagExternalBufferOutput" },
                { Gna2MemoryTagScratch, "Gna2MemoryTagScratch" },
                { Gna2MemoryTagState, "Gna2MemoryTagState" },
        };
        if (!isTagSet) {
            return "Gna2MemoryTag_NotSet_";
        }
        auto f = tm.find(tag);
        if (f != tm.end()) {
            return f->second;
        }
        return "Gna2MemoryTag_" + std::to_string(tag) + "_";
    }
    std::pair<bool, size_t> getOffset(void* offset) const {
        std::pair<bool, size_t> v;
        v.first = offset >= ptr && offset < static_cast<uint8_t*>(ptr) + sizeGranted;
        v.second = v.first ? static_cast<uint8_t*>(offset) - static_cast<uint8_t*>(ptr) : 0;
        return v;
    }

private:
    Gna2MemoryTag tag;
    bool isTagSet = false;
};
typedef std::list<GnaAllocation> GnaAllAllocations;

/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper {
    static std::mutex acrossPluginsSync;
    static std::string decoratedGnaLibVersion() {
        static std::string gnaLibraryVersion{ ", GNA library version: " + GNADeviceHelper::GetGnaLibraryVersion() };
        return gnaLibraryVersion;
    }

    std::string modeOfOperation = "default";
    GnaAllAllocations allAllocations;
    uint32_t nGnaDeviceIndex = 0;
    bool swExactMode = false;
    Gna2DeviceVersion detectedGnaDevVersion = Gna2DeviceVersionSoftwareEmulation;
    std::string executionTarget;
    std::string compileTarget;
    bool useDeviceEmbeddedExport = false;
    Gna2DeviceVersion exportGeneration = Gna2DeviceVersionEmbedded1_0;

    static const uint32_t TotalGna2InstrumentationPoints = 2;
    Gna2InstrumentationPoint gna2InstrumentationPoints[TotalGna2InstrumentationPoints] = {
        Gna2InstrumentationPointHwTotalCycles,
        Gna2InstrumentationPointHwStallCycles };

    uint64_t instrumentationResults[TotalGna2InstrumentationPoints] = {};
    uint64_t instrumentationTotal[TotalGna2InstrumentationPoints] = {};
    uint32_t instrumentationConfigId = 0;
    std::set<uint32_t> unwaitedRequestIds;
#define MAX_TIMEOUT 500000
    bool isPerformanceMeasuring = false;
    bool deviceOpened = false;

public:
    explicit GNADeviceHelper(std::string executionTargetIn = "",
         std::string compileTargetIn = "",
         bool swExactModeIn = false,
         bool isPerformanceMeasuring = false,
         bool deviceEmbedded = false,
         int deviceVersionParsed = 0) :
         swExactMode(swExactModeIn),
         executionTarget(executionTargetIn),
         compileTarget(compileTargetIn),
         isPerformanceMeasuring(isPerformanceMeasuring),
         nGnaDeviceIndex{selectGnaDevice()},
         useDeviceEmbeddedExport(deviceEmbedded),
         exportGeneration(static_cast<Gna2DeviceVersion>(deviceVersionParsed)) {
        open();
        initGnaPerfCounters();

        // check GNA Library version
        const auto gnaLibVersion = GetGnaLibraryVersion();
    }

    GNADeviceHelper(const GNADeviceHelper&) = delete;
    GNADeviceHelper& operator= (const GNADeviceHelper&) = delete;
    ~GNADeviceHelper() {
        if (deviceOpened) {
            close();
        }
    }

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);
    void tagMemoryRegion(void* memPtr, const GNAPluginNS::memory::rRegion memoryTag);

    static bool isGnaLibVersionSupportGna3();

    void setUpActiveList(unsigned req_config_id, uint32_t layerIndex, uint32_t* ptr_active_indices, uint32_t num_active_indices);
    uint32_t propagate(const uint32_t requestConfigId, Gna2AccelerationMode gna2AccelerationMode);
    uint32_t createModel(Gna2Model& gnaModel) const;
    void releaseModel(const uint32_t model_id);
    uint32_t createRequestConfig(const uint32_t model_id);
    static uint32_t getNumberOfGnaDevices();
    static uint32_t selectGnaDevice();
    static bool isGnaHw(const Gna2DeviceVersion dev) {
        return Gna2DeviceVersionSoftwareEmulation != dev;
    }
    bool hasGnaHw() const {
        return isGnaHw(detectedGnaDevVersion);
    }
    static bool isUpTo20HwGnaDevice(const Gna2DeviceVersion dev) {
        return dev <= Gna2DeviceVersion2_0 && isGnaHw(dev);
    }
    bool enforceLegacyCnnNeeded() const;
    static void checkGna2Status(Gna2Status status, const std::string& from);
    static void checkGna2Status(Gna2Status status, const Gna2Model& gnaModel);
    GnaWaitStatus wait(uint32_t id, int64_t millisTimeout = MAX_TIMEOUT);

    struct DumpResult {
        Gna2ModelSueCreekHeader header;
        std::shared_ptr<void> model;
    };

    const void * dumpXNNROPtr = nullptr;
    uint32_t dumpXNNROSize = 0;

    DumpResult dumpXnn(const uint32_t modelId);

    void dumpXnnForDeviceVersion(const uint32_t modelId,
        std::ostream & outStream,
        Gna2DeviceVersion targetDeviceVersion);

    void dumpTLVForDeviceVersion(const uint32_t modelId, std::ostream& outStream,
        Gna2DeviceVersion targetDeviceVersion, uint32_t input_size, uint32_t output_size,
        float inSF, float outSF);

    void free(void * ptr);

    void updateGnaPerfCounters();
    void getGnaPerfCounters(std::map<std::string,
                        InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters);
    static std::string GetGnaLibraryVersion();
    std::string getEffectiveGnaCompileTarget() const;
    std::string GetCompileTarget() const;

    // used for MODEL_DUMP filename
    void AppendOperationMode(std::string in) {
        modeOfOperation += in;
    }

 private:
    void open();

    void close();
    static std::string getGnaLibraryVersionPrivate();
    static const std::map <Gna2ItemType, const std::string> errorTypes;
    static const std::map <Gna2ErrorType, const std::string> errorReasons;
    static const std::map <Gna2OperationType, const std::string> operationTypes;
    static const std::map <const std::pair<Gna2OperationType, int32_t>, const std::string > operandTypes;

    static void enforceLegacyCnns(Gna2Model& gnaModel);
    static void enforceLegacyCnnsWhenNeeded(Gna2Model& gnaModel);
    static Gna2DeviceVersion parseTarget(const std::string& target);
    Gna2DeviceVersion parseDeclaredTarget(std::string target, const bool execTarget) const;
    Gna2DeviceVersion getDefaultTarget() const;
    Gna2DeviceVersion getTargetDevice(bool execTarget) const;

    void createVirtualDevice(Gna2DeviceVersion devVersion, std::string purpose = "");
    void updateGnaDeviceVersion();

    void initGnaPerfCounters() {
        std::unique_lock<std::mutex> lockGnaCalls{ acrossPluginsSync };
        const auto status = Gna2InstrumentationConfigCreate(TotalGna2InstrumentationPoints,
            gna2InstrumentationPoints,
            instrumentationResults,
            &instrumentationConfigId);
        checkGna2Status(status, "Gna2InstrumentationConfigCreate");
    }
};  // NOLINT
