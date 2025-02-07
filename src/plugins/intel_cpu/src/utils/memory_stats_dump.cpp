// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS
#    include "memory_stats_dump.hpp"

#    include <filesystem>
#    include <fstream>

#    include "debug_capabilities.h"

namespace ov::intel_cpu {

static void dumpStatistics(std::ostream& os,
                           const std::string& network_name,
                           std::deque<CompiledModel::GraphGuard>& graphs,
                           const SocketsWeights& weights_cache) {
    os << "Memory stats for model name: " << network_name << "\n";
    if (graphs.empty()) {
        os << "No graphs were found\n";
        return;
    }
    for (auto&& graph : graphs) {
        CompiledModel::GraphGuard::Lock graph_lock{graph};
        os << "Memory stats for graph name: " << graph_lock._graph.GetName() << "\n\n";
        auto ctx = graph_lock._graph.getGraphContext();
        auto&& statistics = ctx->getAuxiliaryNetworkMemoryControl()->dumpStatistics();

        for (auto&& stat : statistics) {
            os << "Memory control ID: " << stat.first << "\n";
            for (auto&& item : stat.second) {
                os << item << "\n";
            }
        }

        auto& scratchpads = ctx->getScratchPads();
        for (size_t i = 0; i < scratchpads.size(); ++i) {
            os << "Scratchpad " << i << " size: " << scratchpads[i]->size() << " bytes\n\n";
        }
    }
    os << "Weights cache statistics\n";
    auto weights_statistics = weights_cache.dumpStatistics();
    for (auto&& item : weights_statistics) {
        os << "Socket ID: " << item.first << "\n";
        os << "Total size: " << item.second.total_size << " bytes\n";
        os << "Total memory objects: " << item.second.total_memory_objects << "\n";
    }
}

static void dumpStatisticsCSV(std::ofstream& os,
                              const std::string& network_name,
                              std::deque<CompiledModel::GraphGuard>& graphs,
                              const SocketsWeights& weights_cache) {
    for (auto&& graph : graphs) {
        CompiledModel::GraphGuard::Lock graph_lock{graph};
        os << "Memory stats for graph name: " << graph_lock._graph.GetName() << ";;;;;;\n";
        auto ctx = graph_lock._graph.getGraphContext();
        auto&& statistics = ctx->getAuxiliaryNetworkMemoryControl()->dumpStatistics();

        for (auto&& stat : statistics) {
            os << "Memory control ID: " << stat.first << ";;;;;;\n";
            os << "Record name;Total regions [-];Total unique blocks [-];Total size [bytes];Optimal total size "
                  "[bytes];Max region size [bytes]\n";

            for (auto&& item : stat.second) {
                os << item.id << ";" << item.total_regions << ";" << item.total_unique_blocks << ";" << item.total_size
                   << ";" << item.optimal_total_size << ";" << item.max_region_size << ";\n";
            }
        }
        os << ";;;;;;\n";
        os << "Scratchpad stats;;;;;;\n";

        os << "Scratchpad ID;Size [bytes];;;;;\n";

        auto& scratchpads = ctx->getScratchPads();
        for (size_t i = 0; i < scratchpads.size(); ++i) {
            os << i << ";" << scratchpads[i]->size() << ";;;;;\n";
        }
    }
    auto weights_statistics = weights_cache.dumpStatistics();
    if (!weights_statistics.empty()) {
        os << ";;;;;;\n";
        os << "Weights cache statistics;;;;;;\n";
        os << "Socket ID;Total size [bytes];Total memory objects [-];;;\n";
    }

    for (auto&& item : weights_statistics) {
        os << item.first << ";" << item.second.total_size << ";" << item.second.total_memory_objects << ";;;;;\n";
    }
}

void dumpMemoryStats(const DebugCapsConfig& conf,
                     const std::string& network_name,
                     std::deque<CompiledModel::GraphGuard>& graphs,
                     const SocketsWeights& weights_cache) {
    if (graphs.empty()) {
        return;
    }

    if (conf.memoryStatisticsDumpPath.empty()) {
        return;
    }

    if (conf.memoryStatisticsDumpPath == "cout") {
        dumpStatistics(std::cout, network_name, graphs, weights_cache);
        return;
    }

    std::filesystem::path file_path = conf.memoryStatisticsDumpPath;
    if (".csv" == file_path.extension()) {
        file_path.replace_filename(file_path.stem().string() + "_" + network_name + file_path.extension().string());

        std::ofstream csv_output(file_path);
        if (!csv_output.is_open()) {
            OPENVINO_THROW("Cannot open file for writing: ", file_path);
        }

        dumpStatisticsCSV(csv_output, network_name, graphs, weights_cache);
    } else {
        OPENVINO_THROW("Unsupported memory stats output. Should be '*.csv' or 'cout'. Got ", file_path.filename());
    }
}

}  // namespace ov::intel_cpu
#endif  // CPU_DEBUG_CAPS
