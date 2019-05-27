// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <utility>

#include <mvnc.h>

#include <myriad_config.h>

namespace vpu {
namespace MyriadPlugin {

struct GraphDesc {
    ncGraphHandle_t *_graphHandle = nullptr;

    ncTensorDescriptor_t _inputDesc = {};
    ncTensorDescriptor_t _outputDesc = {};

    ncFifoHandle_t *_inputFifoHandle = nullptr;
    ncFifoHandle_t *_outputFifoHandle = nullptr;
};

struct DeviceDesc {
    int _executors = 0;
    int _maxExecutors = 0;
    ncDevicePlatform_t _platform = UNKNOWN_PLATFORM;
    int _deviceIdx = -1;
    ncDeviceHandle_t *_deviceHandle = nullptr;

    bool isBooted() const {
        return _deviceHandle != nullptr;
    }
    bool isEmpty() const {
        return _executors == 0;
    }
    bool isAvailable() const {
        return _executors < _maxExecutors;
    }
};

typedef std::shared_ptr<DeviceDesc> DevicePtr;


class MyriadExecutor {
    Logger::Ptr _log;
    unsigned int _numStages = 0;

public:
    MyriadExecutor(bool forceReset, const LogLevel& vpuLogLevel, const Logger::Ptr& log);
    ~MyriadExecutor() = default;

    /**
     * @brief Get myriad device
     * @return Already booted and empty device or new booted device
     */
    DevicePtr openDevice(std::vector<DevicePtr> &devicePool, const std::shared_ptr<MyriadConfig> &config);

    static void closeDevices(std::vector<DevicePtr> &devicePool);

    void allocateGraph(DevicePtr &device,
                       GraphDesc &graphDesc,
                       const std::vector<char> &graphFileContent,
                       const std::pair<const char*, size_t> &graphHeaderDesc,
                       size_t numStages,
                       const char* networkName);

    void deallocateGraph(DevicePtr &device, GraphDesc &graphDesc);

    void queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                        void *result_data, size_t result_bytes);

    void getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes);

    std::string ncStatusToStr(ncGraphHandle_t *graphHandle, ncStatus_t status);

    std::vector<float> getPerfTimeInfo(ncGraphHandle_t *graphHandle);

    void printThrottlingStatus();

    template<typename T>
    std::vector<T> getGraphInfo(
            ncGraphHandle_t* graphHandle,
            int graphOption,
            int numElems) {
        std::vector<T> out(numElems);

        unsigned int infoByteSize = numElems * sizeof(T);
        if (ncGraphGetOption(graphHandle, graphOption, out.data(), &infoByteSize) != NC_OK) {
            out.clear();
        }

        return out;
    }

private:
    /**
     * @brief Try to boot any available device
     * @param configPlatform Boot the selected platform
     */
    ncStatus_t bootNextDevice(std::vector<DevicePtr> &devicePool, const ncDevicePlatform_t &configPlatform, int watchdogInterval);
};

typedef std::shared_ptr<MyriadExecutor> MyriadExecutorPtr;

}  // namespace MyriadPlugin
}  // namespace vpu
