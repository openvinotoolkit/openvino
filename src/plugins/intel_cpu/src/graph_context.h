// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
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

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 std::shared_ptr<SocketsWeights> w_cache,
                 bool isGraphQuantized,
                 std::shared_ptr<std::vector<MultiCachePtr>> rtParamsCache = nullptr,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr,
                 int sub_stream_id = -1,
                 std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister = nullptr);

    const Config& getConfig() const {
        return config;
    }

    std::shared_ptr<SocketsWeights> getWeightsCaches() const {
        return weightsCache;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return (*weightsCache)[m_numa_id];
    }

    std::shared_ptr<std::vector<MultiCachePtr>> getParamsCaches() const {
        return m_rtParamsCache;
    }

    MultiCachePtr getParamsCache() const {
        return (*m_rtParamsCache)[m_numa_id];
    }

    // DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
    //     if (subStreamID < 0)
    //         subStreamID = 0;
    //     if (subStreamID >= numNumaNodes - 1)
    //         subStreamID = numNumaNodes - 1;
    //     return rtScratchPads[subStreamID];
    // }
    DnnlScratchPadPtr getScratchPad() const {
        // if (subStreamID < 0)
        //     subStreamID = 0;
        // if (subStreamID >= numNumaNodes - 1)
        //     subStreamID = numNumaNodes - 1;
        // const int numa_id = m_streamExecutor->get_numa_node_id();
        return rtScratchPads[m_numa_id];
    }

    int getNumaId() const {
        // if (subStreamID < 0)
        //     subStreamID = 0;
        // if (subStreamID >= numNumaNodes - 1)
        //     subStreamID = numNumaNodes - 1;
        // const int numa_id = m_streamExecutor->get_numa_node_id();
        return m_numa_id;
    }

    // const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
    //     return rtScratchPads;
    // }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

    const ov::threading::CPUStreamsExecutor::Ptr& getCPUStreamExecutor() const {
        return cpuStreamExecutor;
    }

    int getNumNumaNodes() const {
        return numNumaNodes;
    }

    const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return memoryStatesRegister;
    }

int getMainNumaNode() const {
        return mainNumaNode;
    }

    int getSubStreamToUse() const {
        return m_sub_steam_id;
        // const int result = *subStreamToUse;
        // (*subStreamToUse)++;
        // if ((*subStreamToUse + 1) >= numNumaNodes) {
        //     *subStreamToUse = -1;
        // }
        // // *subStreamToUse = (*subStreamToUse + 1) % numNumaNodes;
        // return result;
    }

private:
    Config config;  // network-level config

    std::shared_ptr<SocketsWeights> weightsCache;         // per NUMA node caches for sharing weights data

    std::shared_ptr<std::vector<MultiCachePtr>> m_rtParamsCache;

    bool isGraphQuantizedFlag = false;

    std::vector<DnnlScratchPadPtr> rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr m_streamExecutor;   // stream executor for current graph

    ov::threading::CPUStreamsExecutor::Ptr cpuStreamExecutor;   // cpu stream executor for current graph
    std::shared_ptr<int> subStreamToUse;

    int numNumaNodes = 1;

    int mainNumaNode = 0;
    int m_numa_id = 0;
    int m_sub_steam_id = -1;
    std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister;
};

}  // namespace intel_cpu
}  // namespace ov
