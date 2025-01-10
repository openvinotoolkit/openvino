// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS
#    include "memory_stats_dump.hpp"

#    include "debug_capabilities.h"

namespace ov::intel_cpu {

static void dumpToOstream(std::ostream& os,
                          int level,
                          const std::string& network_name,
                          std::deque<CompiledModel::GraphGuard>& graphs,
                          const SocketsWeights& weights_cache) {
    if (level > 0) {
        os << "Memory stats for model name: " << network_name << " " << std::endl;
        if (graphs.empty()) {
            os << "No graphs were found" << std::endl;
            return;
        }
        for (auto&& graph : graphs) {
            CompiledModel::GraphGuard::Lock graph_lock{graph};
            os << "Memory stats for graph name: " << graph_lock._graph.GetName() << std::endl;
            os << std::endl;
            auto ctx = graph_lock._graph.getGraphContext();
            auto&& statistics = ctx->getNetworkMemoryControl()->dumpStatistics();

            for (auto&& stat : statistics) {
                os << "Memory control ID: " << stat.first << std::endl;
                for (auto&& item : stat.second) {
                    os << item << std::endl;
                }
            }

            auto& scratchpads = ctx->getScratchPads();
            for (size_t i = 0; i < scratchpads.size(); ++i) {
                os << "Scratchpad " << i << " size: " << scratchpads[i]->size() << " bytes" << std::endl;
            }
        }
        os << std::endl;
        os << "Weights cache statistics" << std::endl;
        auto weights_statistics = weights_cache.dumpStatistics();
        for (auto&& item : weights_statistics) {
            os << "Socket ID: " << item.first << std::endl;
            os << "Total size: " << item.second.total_size << " bytes" << std::endl;
            os << "Total memory objects: " << item.second.total_memory_objects << std::endl;
        }
    }
}

void dumpMemoryStats(const DebugCapsConfig& conf,
                     const std::string& network_name,
                     std::deque<CompiledModel::GraphGuard>& graphs,
                     const SocketsWeights& weights_cache) {
    if (graphs.empty()) {
        return;
    }

    const auto level = conf.memoryStatisticsDumpLevel;
    if (level == 0) {
        return;
    }

    dumpToOstream(std::cout, level, network_name, graphs, weights_cache);
}

}  // namespace ov::intel_cpu
#endif  // CPU_DEBUG_CAPS