// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "graph_context.h"

#include <utility>

#include "config.h"
#include "memory_control.hpp"
#include "nodes/memory.hpp"

namespace ov::intel_cpu {

GraphContext::GraphContext(Config config,
                           WeightsSharing::Ptr w_cache,
                           bool isGraphQuantized,
                           ov::threading::IStreamsExecutor::Ptr streamExecutor,
                           std::shared_ptr<SubMemoryManager> sub_memory_manager)
    : m_config(std::move(config)),
      m_weightsCache(std::move(w_cache)),
      m_rtParamsCache(std::make_shared<MultiCache>(m_config.rtCacheCapacity)),
      m_isGraphQuantizedFlag(isGraphQuantized),
      m_streamExecutor(std::move(streamExecutor)),
      m_subMemoryManager(std::move(sub_memory_manager)),

      m_memoryStatesRegister(std::make_shared<node::MemoryStatesRegister>()),
      m_auxiliaryNetworkMemoryControl(std::make_shared<NetworkMemoryControl>()),
      m_memoryControl(m_auxiliaryNetworkMemoryControl->createMemoryControlUnit("main")) {
    if (m_streamExecutor) {
        m_cpuStreamExecutor = std::dynamic_pointer_cast<ov::threading::CPUStreamsExecutor>(m_streamExecutor);
        m_numaNodeId = m_cpuStreamExecutor ? std::max(0, m_cpuStreamExecutor->get_numa_node_id()) : 0;
        auto nNumaNodes = get_num_numa_nodes();
        if (m_numNumaNodes < nNumaNodes) {
            m_numNumaNodes = nNumaNodes;
        }
    }
    // primitive/executors can be shared across sub-stream
    // but scratch pad cannot be shared.
    for (int i = 0; i < m_numNumaNodes; i++) {
        m_rtScratchPads.push_back(std::make_shared<DnnlScratchPad>(getEngine(), i));
    }
}

const dnnl::engine& GraphContext::getEngine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

}  // namespace ov::intel_cpu
