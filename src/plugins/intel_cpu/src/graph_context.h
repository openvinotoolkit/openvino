// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

namespace node {
class MemoryStatesRegister;
} // namespace node

class GraphGlobalContext {
public:
    using Ptr  = std::shared_ptr<GraphGlobalContext>;
    using CPtr = std::shared_ptr<const GraphGlobalContext>;

    GraphGlobalContext(const Config& config,
                       WeightsSharing::Ptr w_cache,
                       bool isGraphQuantized,
                       ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr);

    Config config;  // network-level config

    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data

    MultiCachePtr rtParamsCache;     // primitive cache
    DnnlScratchPadPtr rtScratchPad;  // scratch pad

    bool isGraphQuantizedFlag = false;

    std::vector<DnnlScratchPadPtr> rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr streamExecutor;   // stream executor for current graph

    ov::threading::CPUStreamsExecutor::Ptr cpuStreamExecutor;   // cpu stream executor for current graph

    int numNumaNodes = 1;

    std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister;
};

// context which is specific to the current graph
struct GraphLocalContext {
    int level;
};

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr,
                 int level = -1);

    GraphContext(GraphGlobalContext::CPtr global, GraphLocalContext local)
        : global(global),
          local(local) {}

    static const dnnl::engine& getEngine();

    const Config& getConfig() const {
        return global->config;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return global->weightsCache;
    }


    MultiCachePtr getParamsCache() const {
        return global->rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
        if (subStreamID < 0)
            subStreamID = 0;
        if (subStreamID >= global->numNumaNodes - 1)
            subStreamID = global->numNumaNodes - 1;
        return global->rtScratchPads[subStreamID];
    }

    const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return global->rtScratchPads;
    }

    bool isGraphQuantized() const {
        return global->isGraphQuantizedFlag;
    }

    ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return global->cpuStreamExecutor;
    }

    int getNumNumaNodes() const {
        return global->numNumaNodes;
    }

    const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return global->memoryStatesRegister;
    }

    int level() const {
        return local.level;
    }

    // go one level deeper into the context
    GraphContext::Ptr down() const {
        return std::make_shared<GraphContext>(global, GraphLocalContext{local.level + 1});
    }

private:
    GraphGlobalContext::CPtr global;
    GraphLocalContext local;
};

}  // namespace intel_cpu
}  // namespace ov
