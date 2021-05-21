// Copyright (C) 2018-2021 Intel Corporation
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
#include "myriad_mvnc_wrapper.h"

#include <ie_parameter.hpp>

#include <myriad_config.h>

namespace vpu {
namespace MyriadPlugin {

struct GraphDesc {
    ncGraphHandle_t *_graphHandle = nullptr;
    std::string _name;

    ncTensorDescriptor_t _inputDesc = {};
    ncTensorDescriptor_t _outputDesc = {};

    ncFifoHandle_t *_inputFifoHandle = nullptr;
    ncFifoHandle_t *_outputFifoHandle = nullptr;
};

struct DeviceDesc {
    int _graphNum = 0;
    int _maxGraphNum = 0;
    std::string _name;
    ncDevicePlatform_t _platform = NC_ANY_PLATFORM;
    ncDeviceProtocol_t _protocol = NC_ANY_PROTOCOL;

    int _deviceIdx = -1;
    ncDeviceHandle_t *_deviceHandle = nullptr;

    bool isBooted() const {
        return _deviceHandle != nullptr;
    }
    bool isEmpty() const {
        return _graphNum == 0;
    }
    bool isNotFull() const {
        return _graphNum < _maxGraphNum;
    }

    bool isSuitableForConfig(const MyriadConfig& config) const {
        bool isSuitableByName = true;
        if (!config.deviceName().empty()) {
            isSuitableByName = config.deviceName() == _name;
        }

        return isSuitableByName &&
                ((config.platform() == NC_ANY_PLATFORM) || (_platform == config.platform())) &&
                ((config.protocol() == NC_ANY_PROTOCOL) || (_protocol == config.protocol()));
    }

    Platform revision() const {
        VPU_THROW_UNLESS(_platform != NC_ANY_PLATFORM, "Cannot get a revision from not booted device");
        return _platform == NC_MYRIAD_2 ? Platform::MYRIAD_2 : Platform::MYRIAD_X;
    }
};

typedef std::shared_ptr<DeviceDesc> DevicePtr;


class MyriadExecutor {
    Logger::Ptr _log;
    std::shared_ptr<IMvnc> _mvnc;
    unsigned int _numStages = 0;

public:
    MyriadExecutor(bool forceReset, std::shared_ptr<IMvnc> mvnc,
                   const LogLevel& vpuLogLevel, const Logger::Ptr& log);
    ~MyriadExecutor() = default;

    /**
     * @brief Get myriad device
     * @return Already booted and empty device or new booted device
     */
    DevicePtr openDevice(std::vector<DevicePtr> &devicePool, const MyriadConfig& config);

    static void closeDevices(std::vector<DevicePtr> &devicePool, std::shared_ptr<IMvnc> mvnc);

    void allocateGraph(DevicePtr &device,
                       GraphDesc &graphDesc,
                       const std::vector<char> &graphFileContent,
                       const std::pair<const char*, size_t> &graphHeaderDesc,
                       size_t numStages,
                       const std::string & networkName,
                       int executors);

    void deallocateGraph(DevicePtr &device, GraphDesc &graphDesc);

    void queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                        void *result_data, size_t result_bytes);

    void getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes);

    static std::string ncStatusToStr(ncGraphHandle_t *graphHandle, ncStatus_t status);

    std::vector<float> getPerfTimeInfo(ncGraphHandle_t *graphHandle);

    void printThrottlingStatus();

    static float GetThermal(const DevicePtr& device);

    template<typename T>
    static std::vector<T> getGraphInfo(
            ncGraphHandle_t* graphHandle,
            ncGraphOption_t graphOption,
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
     * @brief Try to boot any available device that suitable for selected platform and protocol
     * @param configPlatform Boot the selected platform
     * @param configProtocol Boot device with selected protocol
     */
    ncStatus_t bootNextDevice(std::vector<DevicePtr> &devicePool,
                              const MyriadConfig& config);
};

typedef std::shared_ptr<MyriadExecutor> MyriadExecutorPtr;

}  // namespace MyriadPlugin
}  // namespace vpu
