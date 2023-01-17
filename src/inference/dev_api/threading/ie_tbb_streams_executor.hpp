// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ie_api.h"
#include "ie_parallel.hpp"
#include "threading/ie_istreams_executor.hpp"

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
namespace InferenceEngine {
/**
 * @class TBBStreamsExecutor
 * @brief CPU Streams executor implementation. Use TBB thread pool to run tasks
 */
class INFERENCE_ENGINE_API_CLASS(TBBStreamsExecutor) : public IStreamsExecutor {
public:
    using Ptr = std::shared_ptr<TBBStreamsExecutor>;
    explicit TBBStreamsExecutor(const Config& config = {});
    ~TBBStreamsExecutor() override;
    void run(Task task) override;
    void Execute(Task task) override;
    int GetStreamId() override;
    int GetNumaNodeId() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}  // namespace InferenceEngine
#endif
