// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_cpu_streams_executor.hpp"

#include <atomic>
#include <cassert>
#include <climits>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <openvino/itt.hpp>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ie_parallel_custom_arena.hpp"
#include "ie_system_conf.h"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "threading/ie_executor_manager.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_thread_affinity.hpp"
#include "threading/ie_thread_local.hpp"

using namespace openvino;

namespace InferenceEngine {
struct CPUStreamsExecutor::Impl : public ov::threading::CPUStreamsExecutor {
    Impl(const InferenceEngine::IStreamsExecutor::Config& config) : ov::threading::CPUStreamsExecutor(config) {}
};

int CPUStreamsExecutor::GetStreamId() {
    return _impl->get_stream_id();
}

int CPUStreamsExecutor::GetNumaNodeId() {
    return _impl->get_numa_node_id();
}

CPUStreamsExecutor::CPUStreamsExecutor(const Config& config) : _impl{new Impl(config)} {}

CPUStreamsExecutor::~CPUStreamsExecutor() {}

void CPUStreamsExecutor::Execute(Task task) {
    _impl->execute(std::move(task));
}

void CPUStreamsExecutor::run(Task task) {
    _impl->run(std::move(task));
}

}  // namespace InferenceEngine
