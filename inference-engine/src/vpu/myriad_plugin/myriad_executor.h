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
#include "myriad_infer_router.h"

#include <ie_parameter.hpp>

#include <myriad_config.h>

namespace vpu {
namespace MyriadPlugin {


class MyriadExecutor {
public:
    MyriadExecutor(const DevicePtr& device, const Logger::Ptr& log);
    ~MyriadExecutor() = default;

    void allocateGraph(const std::vector<char> &graphFileContent,
                       const std::pair<const char*, size_t> &graphHeaderDesc,
                       size_t numStages,
                       const std::string& networkName,
                       int executors);

    void deallocateGraph();

    InferFuture sendInferAsync(
        const std::vector<uint8_t>& inTensor, const TensorBuffer& outTensorBuffer);

    std::vector<float> getPerfTimeInfo();

private:
    DevicePtr    m_device;
    Logger::Ptr  m_log;
    GraphDesc    m_graphDesc;
    unsigned int m_numStages = 0;

    MyriadInferRouter::Ptr m_infersRouter;
};

using MyriadExecutorPtr = std::shared_ptr<MyriadExecutor>;

}  // namespace MyriadPlugin
}  // namespace vpu
