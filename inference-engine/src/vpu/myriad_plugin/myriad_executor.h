// Copyright (C) 2018-2020 Intel Corporation
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
#include "myriad_mvnc_wraper.h"
#include "myriad_device_provider.h"

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

class MyriadExecutor {
    Logger::Ptr _log;
    unsigned int _numStages = 0;

public:
    explicit MyriadExecutor(const Logger::Ptr& log);
    ~MyriadExecutor() = default;

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

    std::vector<float> getPerfTimeInfo(ncGraphHandle_t *graphHandle);

    static float GetThermal(const DevicePtr& device);
};

using MyriadExecutorPtr = std::shared_ptr<MyriadExecutor>;

}  // namespace MyriadPlugin
}  // namespace vpu
