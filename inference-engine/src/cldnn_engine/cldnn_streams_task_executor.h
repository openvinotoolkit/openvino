// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <queue>
#include <map>
#include <atomic>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/ie_itask_executor.hpp>
#include "ie_parallel.hpp"
#include "cldnn_graph.h"

namespace CLDNNPlugin {

/* This structure handles an "execution context" - data required to execute an Infer Request.
 * This includes graph (which handles the intermediate data) and arena/observer for the TBB */
struct MultiWorkerTaskContext {
    std::shared_ptr<CLDNNGraph> ptrGraph;
};

/* Class wrapping multiple worker threads that monitors the same queue with Infer Requests. */
class MultiWorkerTaskExecutor : public InferenceEngine::ITaskExecutor {
    static std::atomic<unsigned int> waitingCounter;

public:
    typedef std::shared_ptr<MultiWorkerTaskExecutor> Ptr;

    explicit MultiWorkerTaskExecutor(const std::vector<InferenceEngine::Task>&, std::string name = "Default");

    ~MultiWorkerTaskExecutor();

    void run(InferenceEngine::Task task) override;

    static unsigned  int GetWaitingCounter() { return waitingCounter.load(); }

    static thread_local MultiWorkerTaskContext ptrContext;

    void stop();

private:
    std::vector<std::thread> _threads;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<InferenceEngine::Task> _taskQueue;
    std::atomic<bool> _isStopped;
    std::string _name;
};

};  // namespace CLDNNPlugin
