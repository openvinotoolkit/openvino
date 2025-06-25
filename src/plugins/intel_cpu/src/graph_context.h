// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>

#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "memory_control.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "sub_memory_manager.hpp"
#include "weights_cache.hpp"

namespace ov::intel_cpu {

namespace node {
class MemoryStatesRegister;
}  // namespace node

class MemoryControl;
class NetworkMemoryControl;

class GraphContext {
public:
    using Ptr = std::shared_ptr<GraphContext>;
    using CPtr = std::shared_ptr<const GraphContext>;

    GraphContext(Config config,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr,
                 std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr);

    [[nodiscard]] const Config& getConfig() const {
        return m_config;
    }

    [[nodiscard]] WeightsSharing::Ptr getWeightsCache() const {
        return m_weightsCache;
    }

    [[nodiscard]] MultiCachePtr getParamsCache() const {
        return m_rtParamsCache;
    }

    [[nodiscard]] MultiCachePtr getSnippetsParamsCache() const {
        return m_snippetsParamsCache;
    }

    [[nodiscard]] DnnlScratchPadPtr getScratchPad() const {
        return m_rtScratchPads[m_numaNodeId];
    }

    [[nodiscard]] const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return m_rtScratchPads;
    }

    static const dnnl::engine& getEngine();

    [[nodiscard]] bool isGraphQuantized() const {
        return m_isGraphQuantizedFlag;
    }

    [[nodiscard]] ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return m_cpuStreamExecutor;
    }

    [[nodiscard]] std::shared_ptr<SubMemoryManager> getSubMemory() const {
        return m_subMemoryManager;
    }

    [[nodiscard]] int getNumNumaNodes() const {
        return m_numNumaNodes;
    }

    [[nodiscard]] const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return m_memoryStatesRegister;
    }

    [[nodiscard]] const std::shared_ptr<MemoryControl>& getMemoryControl() const {
        return m_memoryControl;
    }

    [[nodiscard]] const std::shared_ptr<NetworkMemoryControl>& getAuxiliaryNetworkMemoryControl() const {
        return m_auxiliaryNetworkMemoryControl;
    }

    void releaseMemory() const {
        m_auxiliaryNetworkMemoryControl->releaseMemory();
    }

    void allocateMemory() const {
        for (const auto& controlUnit : m_auxiliaryNetworkMemoryControl->controlUnits()) {
            if (!controlUnit->allocated()) {
                controlUnit->allocateMemory();
            }
        }
    }

private:
    // model-level config
    Config m_config;
    // per NUMA node caches for sharing weights data
    WeightsSharing::Ptr m_weightsCache;
    // primitive cache
    MultiCachePtr m_rtParamsCache;
    MultiCachePtr m_snippetsParamsCache;
    // global scratch pad
    DnnlScratchPadPtr m_rtScratchPad;

    bool m_isGraphQuantizedFlag = false;
    // scratch pad per sub-stream
    std::vector<DnnlScratchPadPtr> m_rtScratchPads;
    // stream executor for current graph
    ov::threading::IStreamsExecutor::Ptr m_streamExecutor;
    // cpu stream executor for current graph
    ov::threading::CPUStreamsExecutor::Ptr m_cpuStreamExecutor;
    // numa submemory manager
    std::shared_ptr<SubMemoryManager> m_subMemoryManager;

    int m_numNumaNodes = 1;
    int m_numaNodeId = 0;

    std::shared_ptr<node::MemoryStatesRegister> m_memoryStatesRegister;
    // auxiliary object to allow creating additional memory control objects if the main one cannot be used
    // i.e. fallback graph for dynamic in-place
    std::shared_ptr<NetworkMemoryControl> m_auxiliaryNetworkMemoryControl;
    // main memory control object, which is supposed to be globally reused
    MemoryControl::Ptr m_memoryControl;
};

}  // namespace ov::intel_cpu
