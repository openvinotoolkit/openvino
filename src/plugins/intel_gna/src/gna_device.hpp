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

#include "gna2_model_export_helper.hpp"
#include "memory/gna_mem_requests.hpp"

#include "gna2-common-api.h"
#include "gna2-inference-api.h"
#include "gna2-instrumentation-api.h"

#include "gna2-memory-api.h"
#include "gna2-model-api.h"
#include "gna2-model-export-api.h"
#include "gna2-model-suecreek-header.h"

#include "gna_device_allocation.hpp"
#include "gna_device_interface.hpp"

/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper : public GNAPluginNS::GNADevice {
    static std::mutex acrossPluginsSync;
    static std::string decoratedGnaLibVersion() {
        static std::string gnaLibraryVersion{ ", GNA library version: " + GNADeviceHelper::GetGnaLibraryVersion() };
        return gnaLibraryVersion;
    }
    std::string modeOfOperation = "default";
    GnaAllocations allAllocations;
    uint32_t nGnaDeviceIndex = 0;
    bool swExactMode = false;
    Gna2DeviceVersion detectedGnaDevVersion = Gna2DeviceVersionSoftwareEmulation;
    std::string executionTarget;
    std::string compileTarget;
    bool useDeviceEmbeddedExport = false;
    uint32_t maxLayersCount_ = 0;

    static const uint32_t TotalGna2InstrumentationPoints = 2;
    Gna2InstrumentationPoint gna2InstrumentationPoints[TotalGna2InstrumentationPoints] = {
        Gna2InstrumentationPointHwTotal,
        Gna2InstrumentationPointHwStall};

    uint64_t instrumentationResults[TotalGna2InstrumentationPoints] = {};
    uint64_t instrumentationTotal[TotalGna2InstrumentationPoints] = {};
    uint32_t instrumentationConfigId = 0;
    std::set<uint32_t> unwaitedRequestIds;
#define MAX_TIMEOUT 500000
    bool isPerformanceMeasuring = false;
    bool deviceOpened = false;

    bool debugLogEnabled = false;
    uint64_t debugLogIndexRequestEnqueue = 0;
    uint64_t debugLogIndexRequestWait = 0;
    static constexpr const char* kDumpExt = ".bin";
    static constexpr const char* kDumpDelimiter = ".";

public:
    explicit GNADeviceHelper(std::string executionTargetIn = "",
                             std::string compileTargetIn = "",
                             bool swExactModeIn = false,
                             bool isPerformanceMeasuring = false,
                             bool deviceEmbedded = false);

    GNADeviceHelper(const GNADeviceHelper&) = delete;
    GNADeviceHelper& operator= (const GNADeviceHelper&) = delete;
    GNADeviceHelper(GNADeviceHelper&&) = delete;
    GNADeviceHelper& operator=(GNADeviceHelper&&) = delete;
    ~GNADeviceHelper() override;

    void enableDiagnostics();

    /**
     * @brief Dump raw memory of each GNA allocation to files
     * @param idx index to be appended to the file name
     * @param infix File name would a form of <idx><tagName><kDumpDelimiter><infix><kDumpExt>
     */
    void dumpAllAllocations(uint64_t idx, const std::string& infix) const;

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);
    void tagMemoryRegion(void* memPtr, const GNAPluginNS::memory::rRegion memoryTag);

    void releaseModel(const uint32_t model_id);
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
    static std::string checkGna2Status(Gna2Status status, const std::string& from, bool returnInsteadThrow = false);
    static void checkGna2Status(Gna2Status status, const Gna2Model& gnaModel);

    struct DumpResult {
        Gna2ModelSueCreekHeader header;
        std::shared_ptr<void> model;
    };

    const void * dumpXNNROPtr = nullptr;
    uint32_t dumpXNNROSize = 0;

    DumpResult dumpXnn(const uint32_t modelId);

    void dumpTLVForDeviceVersion(const uint32_t modelId,
                                 std::ostream& outStream,
                                 const std::vector<GnaEndpoint>& inputsContainer,
                                 const std::vector<GnaEndpoint>& outputsContainer);

    void free(void * ptr);

    void updateGnaPerfCounters();
    void getGnaPerfCounters(std::map<std::string,
                        InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters);
    static std::string GetGnaLibraryVersion();
    std::string GetCompileTarget() const;

    const GnaAllocations& getAllAllocations() const {
        return allAllocations;
    }

    /**
     * @see GNADevice::createModel()
     */
    uint32_t createModel(Gna2Model& gnaModel) const override;

    /**
     * @see GNADevice::createRequestConfig()
     */
    uint32_t createRequestConfig(const uint32_t modelID) const override;

    /**
     * @see GNADevice::enqueueRequest()
     */
    uint32_t enqueueRequest(const uint32_t requestConfigID, const Gna2AccelerationMode gna2AccelerationMode) override;

    /**
     * @see GNADevice::waitForRequest()
     */
    GNAPluginNS::RequestStatus waitForRequest(uint32_t requestID, int64_t timeoutMilliseconds = MAX_TIMEOUT) override;

    /**
     * @see GNADevice::maxLayersCount()
     */
     uint32_t maxLayersCount() const override;

private:
    void open();

    void close();
    uint32_t retrieveMaxLayersCount();

    static std::string getGnaLibraryVersionPrivate();
    static const std::map <Gna2ItemType, const std::string> errorTypes;
    static const std::map <Gna2ErrorType, const std::string> errorReasons;
    static const std::map <Gna2OperationType, const std::string> operationTypes;
    static const std::map <const std::pair<Gna2OperationType, int32_t>, const std::string > operandTypes;

    static void enforceLegacyCnns(Gna2Model& gnaModel);
    static void enforceLegacyCnnsWhenNeeded(Gna2Model& gnaModel);
    static Gna2DeviceVersion parseTarget(const std::string& target);
    Gna2DeviceVersion getDefaultTarget() const;
    Gna2DeviceVersion getTargetDevice(bool execTarget) const;

    void createVirtualDevice(Gna2DeviceVersion devVersion);
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
