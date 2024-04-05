// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "dnnl_types.h"
#include "graph_context.h"
#include "nodes/memory.hpp"

namespace ov {
namespace intel_cpu {

GraphContext::GraphContext(const Config& config,
                           std::shared_ptr<SocketsWeights> w_cache,
                           bool isGraphQuantized,
                           std::shared_ptr<std::vector<MultiCachePtr>> rtParamsCache,
                           ov::threading::IStreamsExecutor::Ptr streamExecutor,
                           int sub_stream_id,
                           std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister)
    : config(config),
      weightsCache(std::move(w_cache)),
      isGraphQuantizedFlag(isGraphQuantized),
      m_streamExecutor(std::move(streamExecutor)),
      m_numa_id(sub_stream_id + 1),
      m_sub_steam_id(sub_stream_id),
      memoryStatesRegister(memoryStatesRegister ? memoryStatesRegister : std::make_shared<node::MemoryStatesRegister>()) {
    // primitive/executors can be shared across sub-stream
    // but scratch pad cannot be shared.
    numNumaNodes = 1;
    if (m_streamExecutor) {
        cpuStreamExecutor = std::dynamic_pointer_cast<ov::threading::CPUStreamsExecutor>(m_streamExecutor);
        auto nNumaNodes = get_num_numa_nodes();
        if (numNumaNodes < nNumaNodes)
            numNumaNodes = nNumaNodes;

        // mainNumaNode = m_streamExecutor->get_numa_node_id();
        mainNumaNode = m_streamExecutor->get_socket_id();
    }

    // if (!rtParamsCache)
    m_rtParamsCache = std::make_shared<std::vector<MultiCachePtr>>(numNumaNodes, std::make_shared<MultiCache>(config.rtCacheCapacity));
    // else
    //     m_rtParamsCache = rtParamsCache;

    subStreamToUse = std::make_shared<int>(numNumaNodes - 2);

    for (int i = 0; i < numNumaNodes; i++) {
        rtScratchPads.push_back(std::make_shared<DnnlScratchPad>(getEngine(), i));
    }
}

const dnnl::engine& GraphContext::getEngine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

}   // namespace intel_cpu
}   // namespace ov
