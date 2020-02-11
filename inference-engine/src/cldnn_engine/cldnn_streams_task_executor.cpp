// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <map>
#include <atomic>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include <future>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/ie_itask_executor.hpp>
#include "ie_parallel.hpp"
#include "cldnn_streams_task_executor.h"

namespace CLDNNPlugin {
std::atomic<unsigned int> MultiWorkerTaskExecutor::waitingCounter(0u);

thread_local MultiWorkerTaskContext MultiWorkerTaskExecutor::ptrContext;

MultiWorkerTaskExecutor::MultiWorkerTaskExecutor(const std::vector<InferenceEngine::Task>& init_tasks, std::string name) :
        _isStopped(false), _name(name) {
    std::vector<std::packaged_task<void()>> initTasks;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < init_tasks.size(); t++) {
        initTasks.emplace_back([&init_tasks, t] {init_tasks[t]();});
        futures.emplace_back(initTasks.back().get_future());
    }
    for (int t = 0; t < init_tasks.size(); t++) {
        _threads.emplace_back([&, t] {
            // initialization (no contention, every worker thread is doing it's own task)
            initTasks[t]();

            while (!_isStopped) {
                InferenceEngine::Task currentTask = nullptr;
                {  // waiting for the new task or for stop signal
                    std::unique_lock<std::mutex> lock(_queueMutex);
                    _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                    if (!_taskQueue.empty()) {
                        currentTask = std::move(_taskQueue.front());
                        _taskQueue.pop();
                    }
                }
                if (currentTask) {
                    waitingCounter--;
                    currentTask();
                }
            }
            // WA as destroying last cl::Context in thread exit causes deadlock
            MultiWorkerTaskExecutor::ptrContext.ptrGraph = nullptr;
        });
    }
    for (auto&& f : futures)
        f.wait();
    for (auto&& f : futures) {
        try {
            f.get();
        } catch(...) {
            stop();
            throw;
        }
    }
}

void MultiWorkerTaskExecutor::stop() {
    _isStopped = true;
    _queueCondVar.notify_all();
    for (auto& thread : _threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

MultiWorkerTaskExecutor::~MultiWorkerTaskExecutor() {
    stop();
}

void MultiWorkerTaskExecutor::run(InferenceEngine::Task task) {
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _taskQueue.push(std::move(task));
        waitingCounter++;
    }
    _queueCondVar.notify_one();
}

};  // namespace CLDNNPlugin
