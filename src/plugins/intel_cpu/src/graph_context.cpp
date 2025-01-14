// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "graph_context.h"

#include "dnnl_types.h"
#include "memory_control.hpp"
#include "nodes/memory.hpp"

namespace ov {
namespace intel_cpu {

GraphContext::GraphContext(const Config& config,
                           WeightsSharing::Ptr w_cache,
                           bool isGraphQuantized,
                           ov::threading::IStreamsExecutor::Ptr streamExecutor,
                           std::shared_ptr<SubMemoryManager> sub_memory_manager)
    : config(config),
      weightsCache(std::move(w_cache)),
      isGraphQuantizedFlag(isGraphQuantized),
      streamExecutor(streamExecutor),
      subMemoryManager(std::move(sub_memory_manager)),
      memoryStatesRegister(std::make_shared<node::MemoryStatesRegister>()),
      networkMemoryControl(std::make_shared<NetworkMemoryControl>()) {
    rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);
    // primitive/executors can be shared across sub-stream
    // but scratch pad cannot be shared.
    numNumaNodes = 1;
    if (streamExecutor) {
        cpuStreamExecutor = std::dynamic_pointer_cast<ov::threading::CPUStreamsExecutor>(streamExecutor);
        numaNodeId = cpuStreamExecutor ? cpuStreamExecutor->get_numa_node_id() : 0;
        auto nNumaNodes = get_num_numa_nodes();
        if (numNumaNodes < nNumaNodes)
            numNumaNodes = nNumaNodes;
    }
    for (int i = 0; i < numNumaNodes; i++) {
        rtScratchPads.push_back(std::make_shared<DnnlScratchPad>(getEngine(), i));
    }
}

const dnnl::engine& GraphContext::getEngine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

}  // namespace intel_cpu
}  // namespace ov
