// Copyright (C) 2018-2019 Intel Corporation
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
#include <cpp_interfaces/ie_task_executor.hpp>
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

    explicit MultiWorkerTaskExecutor(const std::vector<InferenceEngine::Task::Ptr>&, std::string name = "Default");

    ~MultiWorkerTaskExecutor();

    /**
    * @brief Adds task for execution and notifies one of the working threads about the new task.
    * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
    * @param task - shared pointer to the task
    *  @return true if succeed to add task, otherwise - false
    */
    bool startTask(InferenceEngine::Task::Ptr task) override;

    static unsigned  int GetWaitingCounter() { return waitingCounter.load(); }

    static thread_local MultiWorkerTaskContext ptrContext;

private:
    std::vector<std::thread> _threads;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<InferenceEngine::Task::Ptr> _taskQueue;
    std::atomic<bool> _isStopped;
    std::string _name;
    std::atomic<int> _initCount;
};

};  // namespace CLDNNPlugin
