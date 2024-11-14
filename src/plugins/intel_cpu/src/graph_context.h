// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "sub_memory_manager.hpp"
#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

namespace node {
class MemoryStatesRegister;
} // namespace node

class NetworkMemoryControl;

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr,
                 std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr);

    const Config& getConfig() const {
        return config;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }


    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
        if (subStreamID < 0)
            subStreamID = 0;
        if (subStreamID >= numNumaNodes - 1)
            subStreamID = numNumaNodes - 1;
        return rtScratchPads[subStreamID];
    }

    const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return rtScratchPads;
    }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

    ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return cpuStreamExecutor;
    }

    std::shared_ptr<SubMemoryManager> getSubMemory() const {
        return subMemoryManager;
    }

    int getNumNumaNodes() const {
        return numNumaNodes;
    }

    const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return memoryStatesRegister;
    }

    const std::shared_ptr<NetworkMemoryControl>& getNetworkMemoryControl() const {
        return networkMemoryControl;
    }

private:
    Config config;  // network-level config

    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;

    std::vector<DnnlScratchPadPtr> rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr streamExecutor;   // stream executor for current graph

    ov::threading::CPUStreamsExecutor::Ptr cpuStreamExecutor;   // cpu stream executor for current graph

    std::shared_ptr<SubMemoryManager> subMemoryManager;

    int numNumaNodes = 1;

    std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister;
    std::shared_ptr<NetworkMemoryControl> networkMemoryControl;
};

}  // namespace intel_cpu
}  // namespace ov
