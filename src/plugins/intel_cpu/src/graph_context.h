// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "sub_memory_manager.hpp"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

namespace node {
class MemoryStatesRegister;
}  // namespace node

class NetworkMemoryControl;

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(Config config,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr,
                 std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr);

    const Config& getConfig() const {
        return m_config;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return m_weightsCache;
    }

    MultiCachePtr getParamsCache() const {
        return m_rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return m_rtScratchPads[m_numaNodeId];
    }

    const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return m_rtScratchPads;
    }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return m_isGraphQuantizedFlag;
    }

    ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return m_cpuStreamExecutor;
    }

    std::shared_ptr<SubMemoryManager> getSubMemory() const {
        return m_subMemoryManager;
    }

    int getNumNumaNodes() const {
        return m_numNumaNodes;
    }

    const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return m_memoryStatesRegister;
    }

    const std::shared_ptr<NetworkMemoryControl>& getNetworkMemoryControl() const {
        return m_networkMemoryControl;
    }

private:
    Config m_config;  // network-level config

    WeightsSharing::Ptr m_weightsCache;  // per NUMA node caches for sharing weights data

    MultiCachePtr m_rtParamsCache;     // primitive cache
    DnnlScratchPadPtr m_rtScratchPad;  // scratch pad

    bool m_isGraphQuantizedFlag = false;

    std::vector<DnnlScratchPadPtr> m_rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr m_streamExecutor;  // stream executor for current graph

    ov::threading::CPUStreamsExecutor::Ptr m_cpuStreamExecutor;  // cpu stream executor for current graph

    std::shared_ptr<SubMemoryManager> m_subMemoryManager;

    int m_numNumaNodes = 1;
    int m_numaNodeId = 0;

    std::shared_ptr<node::MemoryStatesRegister> m_memoryStatesRegister;
    std::shared_ptr<NetworkMemoryControl> m_networkMemoryControl;
};

}  // namespace intel_cpu
}  // namespace ov
