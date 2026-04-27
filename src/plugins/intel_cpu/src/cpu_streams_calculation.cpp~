// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "config.h"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"

#if (defined(OPENVINO_ARCH_ARM64) && defined(__linux__))
#    include "cpu/aarch64/cpu_isa_traits.hpp"
#else
#    if !defined(OPENVINO_ARCH_RISCV64)
#        include <oneapi/dnnl/dnnl.hpp>

#        include "onednn/dnnl.h"
#    endif
#    include "openvino/runtime/performance_heuristics.hpp"
#endif
#include "cpu_map_scheduling.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace ov;
using namespace ov::threading;

constexpr int INIT_VAL = -100;
constexpr int TP_CPU_LIMIT = 32;

namespace ov::intel_cpu {

namespace ThreadPreferenceConstants {

[[maybe_unused]] constexpr int INT8_EFFICIENCY_THRESHOLD = 4;
[[maybe_unused]] constexpr int FP32_EFFICIENCY_THRESHOLD = 2;

[[maybe_unused]] constexpr float ISA_THRESHOLD_SSE41 = 0.5F;
[[maybe_unused]] constexpr float ISA_THRESHOLD_AVX2 = 1.0F;
[[maybe_unused]] constexpr float ISA_THRESHOLD_VNNI = 2.0F;
[[maybe_unused]] constexpr float ISA_THRESHOLD_AMX = 4.0F;

[[maybe_unused]] constexpr float MEM_TOLERANCE_VERY_HIGH = 50.0F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_HIGH = 4.5F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_MEDIUM_HIGH = 2.5F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_MEDIUM = 1.0F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_MEDIUM_LOW = 0.5F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_LOW = 0.2F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_SECONDARY_LOW = 0.08F;
[[maybe_unused]] constexpr float MEM_TOLERANCE_VERY_LOW = 0.06F;

[[maybe_unused]] constexpr float CONV_RATIO_VERY_HIGH = 0.9F;
[[maybe_unused]] constexpr float CONV_RATIO_HIGH = 0.8F;
[[maybe_unused]] constexpr float CONV_RATIO_MEDIUM = 0.6F;
[[maybe_unused]] constexpr float CONV_RATIO_MEDIUM_LOW = 0.5F;
[[maybe_unused]] constexpr float CONV_RATIO_LOW = 0.46F;
[[maybe_unused]] constexpr float CONV_RATIO_MINIMAL = 0.28F;
[[maybe_unused]] constexpr float CONV_RATIO_VERY_LOW = 0.2F;
[[maybe_unused]] constexpr float CONV_RATIO_ULTRA_LOW = 0.1F;

[[maybe_unused]] constexpr float GEMM_RATIO_HIGH = 0.14F;
[[maybe_unused]] constexpr float GEMM_RATIO_LOW = 0.05F;

[[maybe_unused]] constexpr int ECORE_RATIO_THRESHOLD = 2;

[[maybe_unused]] constexpr int ARM64_THREADS_DEFAULT = 8;
[[maybe_unused]] constexpr int ARM64_THREADS_SVE = 16;

[[maybe_unused]] constexpr int ARM_THREADS_DEFAULT = 4;
[[maybe_unused]] constexpr int ARM_THREADS_HIGH = 8;

[[maybe_unused]] constexpr int APPLE_THREADS_MINIMAL = 1;
[[maybe_unused]] constexpr int APPLE_THREADS_LOW = 2;
[[maybe_unused]] constexpr int APPLE_THREADS_HIGH = 4;

}  // namespace ThreadPreferenceConstants

namespace {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
float get_isa_threshold_multiplier(dnnl::cpu_isa isa) {
    using namespace ThreadPreferenceConstants;
    switch (isa) {
    case dnnl::cpu_isa::sse41:
        return ISA_THRESHOLD_SSE41;
    case dnnl::cpu_isa::avx2:
    case dnnl::cpu_isa::avx512_core:
        return ISA_THRESHOLD_AVX2;
    case dnnl::cpu_isa::avx512_core_vnni:
    case dnnl::cpu_isa::avx2_vnni:
    case dnnl::cpu_isa::avx2_vnni_2:
        return ISA_THRESHOLD_VNNI;
    case dnnl::cpu_isa::avx512_core_amx:
        return ISA_THRESHOLD_AMX;
    default:
        return ISA_THRESHOLD_AVX2;
    }
}
#endif

bool should_use_all_cores_for_latency(int main_cores, int efficient_cores, bool int8_intensive) {
    using namespace ThreadPreferenceConstants;
    const int threshold = int8_intensive ? INT8_EFFICIENCY_THRESHOLD : FP32_EFFICIENCY_THRESHOLD;
    return main_cores * threshold <= efficient_cores;
}

bool should_use_ecores_for_llm(int efficient_cores, int main_cores) {
    using namespace ThreadPreferenceConstants;
    return efficient_cores > ECORE_RATIO_THRESHOLD * main_cores;
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_RISCV64)
bool is_main_core_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs > CONV_RATIO_HIGH;
}

bool is_main_core_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs == 0.0F && tolerance.ratio_compute_convs == 0.0F &&
           tolerance.max_mem_tolerance >= MEM_TOLERANCE_HIGH;
}

bool is_main_core_case_3(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs == 0.0F && tolerance.ratio_compute_convs > 0.0F &&
           tolerance.ratio_compute_convs < 1.0F &&
           static_cast<float>(tolerance.total_light_convs) >
               CONV_RATIO_VERY_HIGH * static_cast<float>(tolerance.total_convs);
}

bool is_main_core_case_4(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs > 0.0F && tolerance.ratio_compute_convs > 0.0F &&
           static_cast<float>(tolerance.total_light_convs) > CONV_RATIO_LOW * static_cast<float>(tolerance.total_convs);
}

bool is_static_partitioner_case_1(const ov::MemBandwidthPressure& tolerance) {
    return tolerance.total_nodes == 0;
}

bool is_static_partitioner_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 && static_cast<float>(tolerance.total_light_convs) >
                                            CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs);
}

bool is_static_partitioner_case_3_with_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 &&
           static_cast<float>(tolerance.total_light_convs) <=
               CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs) &&
           tolerance.ratio_compute_convs + tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_HIGH &&
           tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_LOW && tolerance.ratio_mem_limited_gemms == 0.0F &&
           ((tolerance.ratio_mem_limited_adds < CONV_RATIO_MINIMAL &&
             tolerance.max_mem_tolerance >= MEM_TOLERANCE_VERY_LOW) ||
            tolerance.ratio_compute_convs == 0 || tolerance.ratio_mem_limited_convs == 0);
}

bool is_static_partitioner_case_3_without_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 &&
           static_cast<float>(tolerance.total_light_convs) <=
               CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs) &&
           tolerance.ratio_compute_convs + tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_HIGH &&
           tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_LOW && tolerance.ratio_mem_limited_gemms == 0.0F &&
           tolerance.ratio_mem_limited_adds < CONV_RATIO_MINIMAL &&
           tolerance.max_mem_tolerance >= MEM_TOLERANCE_VERY_LOW;
}

bool is_static_partitioner_case_4_with_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 &&
           (tolerance.max_mem_tolerance > MEM_TOLERANCE_MEDIUM_HIGH ||
            static_cast<float>(tolerance.total_gemms) >= GEMM_RATIO_HIGH * static_cast<float>(tolerance.total_nodes));
}

bool is_static_partitioner_case_4_without_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 &&
           static_cast<float>(tolerance.total_gemms) < GEMM_RATIO_LOW * static_cast<float>(tolerance.total_nodes);
}

bool is_static_partitioner_case_5(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 &&
           static_cast<float>(tolerance.total_light_convs) <=
               CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs) &&
           tolerance.ratio_compute_convs >= CONV_RATIO_VERY_HIGH * tolerance.ratio_mem_limited_convs &&
           tolerance.ratio_compute_convs == 1.0F && tolerance.ratio_mem_limited_adds == 1.0F &&
           static_cast<float>(tolerance.total_heavy_convs) >
               CONV_RATIO_ULTRA_LOW * static_cast<float>(tolerance.total_nodes);
}

void determine_tbb_partitioner_and_threads(Config& config,
                                           const std::vector<std::vector<int>>& proc_type_table,
                                           const ov::MemBandwidthPressure& tolerance,
                                           bool int8_intensive) {
    if (config.tbbPartitioner != TbbPartitioner::NONE) {
        return;
    }

    const bool has_lp_ecores = proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0;

    if (has_lp_ecores && int8_intensive && tolerance.total_convs > 0) {
        if (is_main_core_case_1(tolerance) || is_main_core_case_2(tolerance) || is_main_core_case_3(tolerance) ||
            is_main_core_case_4(tolerance)) {
            config.modelPreferThreadsLatency = proc_type_table[0][MAIN_CORE_PROC];
            config.tbbPartitioner = TbbPartitioner::STATIC;
            return;
        }
    }

    bool static_case_3 = has_lp_ecores ? is_static_partitioner_case_3_with_lp_ecores(tolerance)
                                       : is_static_partitioner_case_3_without_lp_ecores(tolerance);

    bool static_case_4 = has_lp_ecores ? is_static_partitioner_case_4_with_lp_ecores(tolerance)
                                       : is_static_partitioner_case_4_without_lp_ecores(tolerance);

    bool static_case_5 = has_lp_ecores && is_static_partitioner_case_5(tolerance);

    if (is_static_partitioner_case_1(tolerance) || is_static_partitioner_case_2(tolerance) || static_case_3 ||
        static_case_4 || static_case_5) {
        config.tbbPartitioner = TbbPartitioner::STATIC;
    } else {
        config.tbbPartitioner = TbbPartitioner::AUTO;
    }
}
#endif

[[maybe_unused]] bool is_network_compute_limited(const ov::MemBandwidthPressure& tolerance) {
    return tolerance.ratio_compute_convs == ov::MemBandwidthPressure::ALL ||
           tolerance.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL;
}

[[maybe_unused]] bool is_below_isa_threshold(float max_tolerance, float memThresholdAssumeLimitedForISA) {
    return max_tolerance > memThresholdAssumeLimitedForISA;
}

[[maybe_unused]] bool is_below_general_threshold(float max_tolerance) {
    return max_tolerance > ov::MemBandwidthPressure::LIMITED;
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_RISCV64)
bool is_lp_main_core_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 && tolerance.max_mem_tolerance > MEM_TOLERANCE_VERY_HIGH &&
           static_cast<float>(tolerance.total_gemms) < GEMM_RATIO_LOW * static_cast<float>(tolerance.total_nodes);
}

bool is_lp_main_core_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 && tolerance.total_gemms == 1 &&
           tolerance.max_mem_tolerance<MEM_TOLERANCE_MEDIUM_LOW&& static_cast<float>(
               tolerance.total_light_convs)> CONV_RATIO_HIGH *
               static_cast<float>(tolerance.total_convs);
}

bool is_lp_auto_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.max_mem_tolerance < MEM_TOLERANCE_MEDIUM && tolerance.ratio_compute_convs == 0 &&
           tolerance.ratio_mem_limited_convs == 0;
}

bool is_lp_auto_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_compute_convs + tolerance.ratio_mem_limited_convs >= 1.0F &&
           tolerance.ratio_mem_limited_deconvs < 1.0F;
}

bool is_lp_auto_case_3(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 52 && tolerance.ratio_compute_convs > 0 && tolerance.ratio_mem_limited_convs > 0 &&
           tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_LOW;
}

bool is_lp_auto_case_4(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.max_mem_tolerance < MEM_TOLERANCE_SECONDARY_LOW &&
           tolerance.ratio_compute_convs < CONV_RATIO_HIGH && tolerance.ratio_mem_limited_convs < CONV_RATIO_MEDIUM_LOW;
}

bool is_lp_auto_case_5(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_compute_convs > 0 && tolerance.ratio_compute_convs < CONV_RATIO_ULTRA_LOW &&
           tolerance.ratio_mem_limited_convs >= CONV_RATIO_VERY_LOW;
}
#endif

}  // namespace

void sort_table_by_numa_node_id(int current_numa_node, std::vector<std::vector<int>>& proc_type_table) {
    if (proc_type_table.size() > 1) {
        for (size_t i = 1; i < proc_type_table.size(); i++) {
            if (current_numa_node == proc_type_table[i][PROC_NUMA_NODE_ID]) {
                std::rotate(proc_type_table.begin() + 1, proc_type_table.begin() + i, proc_type_table.end());
                break;
            }
        }
    }
}

struct StreamsInfoBuilder {
    const int input_streams;
    const bool input_streams_changed;
    const int input_threads;
    const int input_infer_requests;
    const int model_prefer_threads;
    const bool enable_tensor_parallel;
    const std::string& input_perf_hint;
    const std::set<ov::hint::ModelDistributionPolicy>& hint_model_distribution_policy;
    const std::vector<std::vector<int>>& proc_type_table;

    std::vector<int> stream_info;
    std::vector<std::vector<int>> streams_info_table;
    std::vector<std::vector<int>> proc_socket_table;

    int n_streams = 0;
    int n_threads = 0;
    int n_threads_per_stream = 0;
    int current_socket_id = -1;

    void set_ids(const std::vector<int>& one_proc_info) {
        stream_info[STREAM_NUMA_NODE_ID] = one_proc_info[PROC_NUMA_NODE_ID];
        stream_info[STREAM_SOCKET_ID] = one_proc_info[PROC_SOCKET_ID];
    }

    /**
     * Socket selection priority (when current_socket_id >= 0):
     *   n_mode 3 → same socket as current_socket_id
     *   n_mode 2 → other sockets
     *   n_mode 1 → any socket
     */
    void add_mixed_stream(const std::vector<int>& one_proc_info,
                          const std::vector<std::vector<int>>& one_proc_table,
                          int num_threads,
                          IStreamsExecutor::Config::StreamsMode sub_streams_model,
                          int target_proc) {
        stream_info[PROC_TYPE] = ALL_PROC;
        stream_info[NUMBER_OF_STREAMS] =
            sub_streams_model == IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL ? 1 : -1;
        stream_info[THREADS_PER_STREAM] = num_threads;
        set_ids(one_proc_info);
        streams_info_table.push_back(stream_info);

        stream_info[NUMBER_OF_STREAMS] = 0;
        int total_threads = stream_info[THREADS_PER_STREAM];
        int socket_id = stream_info[STREAM_SOCKET_ID];
        int node_start = one_proc_table.size() == 1 ? 0 : 1;
        auto node_end = static_cast<int>(one_proc_table.size() == 1 ? 1 : one_proc_table.size());

        for (int n_mode = current_socket_id < 0 ? 1 : 3; (n_mode > 0) && (total_threads > 0); n_mode--) {
            for (int n = MAIN_CORE_PROC; (n <= HYPER_THREADING_PROC) && (total_threads > 0); n++) {
                for (int index = node_start; (index < node_end) && (total_threads > 0); index++) {
                    if (!matches_socket_mode(n_mode, socket_id, one_proc_table[index][PROC_SOCKET_ID])) {
                        continue;
                    }
                    if ((0 == one_proc_table[index][n]) || !any_of(target_proc, ALL_PROC, n)) {
                        continue;
                    }
                    stream_info[PROC_TYPE] = n;
                    stream_info[STREAM_NUMA_NODE_ID] = one_proc_table[index][PROC_NUMA_NODE_ID];
                    stream_info[STREAM_SOCKET_ID] = one_proc_table[index][PROC_SOCKET_ID];
                    if (total_threads <= one_proc_table[index][n]) {
                        stream_info[THREADS_PER_STREAM] = total_threads;
                        streams_info_table.push_back(stream_info);
                        return;
                    }
                    stream_info[THREADS_PER_STREAM] = one_proc_table[index][n];
                    streams_info_table.push_back(stream_info);
                    total_threads -= stream_info[THREADS_PER_STREAM];
                }
            }
        }
    }

    void add_single_stream(const std::vector<int>& one_proc_info,
                           const std::vector<std::vector<int>>& one_proc_table,
                           [[maybe_unused]] int num_threads,
                           IStreamsExecutor::Config::StreamsMode sub_streams_model) {
        bool needs_mix = (one_proc_info[PROC_NUMA_NODE_ID] < 0) || (one_proc_info[PROC_SOCKET_ID] < 0) ||
                         ((one_proc_info[MAIN_CORE_PROC] > 0) &&
                          (one_proc_info[MAIN_CORE_PROC] < stream_info[THREADS_PER_STREAM])) ||
                         ((one_proc_info[MAIN_CORE_PROC] == 0) && (one_proc_info[EFFICIENT_CORE_PROC] > 0) &&
                          (one_proc_info[EFFICIENT_CORE_PROC] < stream_info[THREADS_PER_STREAM]));

        if (needs_mix) {
            add_mixed_stream(one_proc_info,
                             one_proc_table,
                             stream_info[THREADS_PER_STREAM],
                             sub_streams_model,
                             ALL_PROC);
        } else {
            stream_info[PROC_TYPE] =
                one_proc_info[MAIN_CORE_PROC] >= stream_info[THREADS_PER_STREAM] ? MAIN_CORE_PROC : EFFICIENT_CORE_PROC;
            stream_info[NUMBER_OF_STREAMS] =
                sub_streams_model == IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL ? 1 : -1;
            set_ids(one_proc_info);
            streams_info_table.push_back(stream_info);
        }
    }

    void distribute_streams_per_node(int proc_type, const std::vector<int>& one_proc_info) {
        if ((one_proc_info[PROC_NUMA_NODE_ID] < 0) && (stream_info[NUMBER_OF_STREAMS] == 1)) {
            add_mixed_stream(one_proc_info,
                             proc_type_table,
                             one_proc_info[ALL_PROC],
                             IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                             proc_type);
            return;
        }
        if (0 == one_proc_info[proc_type]) {
            return;
        }
        if (n_threads_per_stream == -1) {
            stream_info[THREADS_PER_STREAM] = any_of(proc_type, EFFICIENT_CORE_PROC, LP_EFFICIENT_CORE_PROC) ? 2 : 1;
        }
        stream_info[PROC_TYPE] = proc_type;
        set_ids(one_proc_info);
        stream_info[NUMBER_OF_STREAMS] = static_cast<int>(one_proc_info[proc_type] / stream_info[THREADS_PER_STREAM]);
        if (n_streams < stream_info[NUMBER_OF_STREAMS]) {
            stream_info[NUMBER_OF_STREAMS] = n_streams;
        }
        if (stream_info[NUMBER_OF_STREAMS] > 0) {
            streams_info_table.push_back(stream_info);
            n_streams -= stream_info[NUMBER_OF_STREAMS];
        }
    }

    void adjust_threads_per_stream() {
        int count = 0;
        while (true) {
            for (int n_type = MAIN_CORE_PROC; n_type <= HYPER_THREADING_PROC; n_type++) {
                count += static_cast<int>(proc_type_table[0][n_type] / n_threads_per_stream);
            }
            if (count >= n_streams) {
                return;
            }
            count = 0;
            if (n_threads_per_stream > 1) {
                n_threads_per_stream--;
            } else {
                n_streams = n_threads;
                return;
            }
        }
    }

    void build_socket_table() {
        if (proc_type_table.size() == 1) {
            proc_socket_table.push_back(proc_type_table[0]);
            return;
        }
        std::unordered_set<int> socket_id_list(proc_type_table.size());
        for (size_t i = 1; i < proc_type_table.size(); i++) {
            if (!socket_id_list.count(proc_type_table[i][PROC_SOCKET_ID])) {
                proc_socket_table.push_back(proc_type_table[i]);
                socket_id_list.insert(proc_type_table[i][PROC_SOCKET_ID]);
            } else {
                for (auto& row : proc_socket_table) {
                    if (row[PROC_SOCKET_ID] == proc_type_table[i][PROC_SOCKET_ID]) {
                        for (int n = 0; n <= HYPER_THREADING_PROC; n++) {
                            row[n] += proc_type_table[i][n];
                        }
                        if (row[PROC_NUMA_NODE_ID] != proc_type_table[i][PROC_NUMA_NODE_ID]) {
                            row[PROC_NUMA_NODE_ID] = -1;
                        }
                    }
                }
            }
        }
    }

    [[nodiscard]] bool has_tensor_parallel_policy() const {
        return hint_model_distribution_policy.find(ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL) !=
               hint_model_distribution_policy.end();
    }

    [[nodiscard]] bool is_latency_mode() const {
        return ((!input_streams_changed) &&
                (input_perf_hint == ov::util::to_string(ov::hint::PerformanceMode::LATENCY))) ||
               ((input_streams_changed) && (input_streams == 1));
    }

    void handle_latency_with_explicit_threads() {
        if (hint_model_distribution_policy.empty()) {
            n_threads_per_stream = std::min(input_threads, proc_type_table[0][ALL_PROC]);
        } else {
            for (auto& row : proc_socket_table) {
                if (current_socket_id == row[PROC_SOCKET_ID]) {
                    n_threads_per_stream = std::min(input_threads, row[ALL_PROC]);
                }
            }
        }
        if (proc_type_table.size() == 1) {
            if ((n_threads_per_stream > proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = ALL_PROC;
            }
        }
    }

    void handle_latency_tensor_parallel_or_single_socket() {
        if ((proc_type_table.size() == 1) && (model_prefer_threads > 0)) {
            if ((model_prefer_threads == proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                n_threads_per_stream = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
                set_ids(proc_type_table[0]);
            } else if (proc_type_table[0][MAIN_CORE_PROC] == 0) {
                if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
                    stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                    n_threads_per_stream = proc_type_table[0][EFFICIENT_CORE_PROC];
                } else {
                    stream_info[PROC_TYPE] = LP_EFFICIENT_CORE_PROC;
                    n_threads_per_stream = proc_type_table[0][LP_EFFICIENT_CORE_PROC];
                }
                stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
                set_ids(proc_type_table[0]);
            } else {
                stream_info[PROC_TYPE] = ALL_PROC;
                n_threads_per_stream = proc_type_table[0][ALL_PROC] - proc_type_table[0][LP_EFFICIENT_CORE_PROC];
                if (proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0) {
                    n_threads_per_stream = std::max(model_prefer_threads, n_threads_per_stream);
                    n_threads_per_stream = std::min(n_threads_per_stream, proc_type_table[0][ALL_PROC]);
                }
            }
        } else {
            // Fallback for single-socket or tensor-parallel when model_prefer_threads == 0.
            // Ensure n_threads_per_stream is always positive, even on LP-only systems where
            // ALL_PROC == LP_EFFICIENT_CORE_PROC and the original difference would be zero.
            n_threads_per_stream = proc_type_table[0][ALL_PROC] - proc_type_table[0][LP_EFFICIENT_CORE_PROC];
            if (n_threads_per_stream <= 0) {
                n_threads_per_stream = std::max(1, proc_type_table[0][ALL_PROC]);
            }
        }
    }

    void handle_latency_multi_socket() {
        size_t socket_index = 0;
        for (socket_index = 0; socket_index < proc_socket_table.size(); socket_index++) {
            if (proc_socket_table[socket_index][PROC_SOCKET_ID] == current_socket_id) {
                break;
            }
        }
        const std::vector<int>& current_socket_info = proc_socket_table[socket_index];
        n_threads_per_stream = model_prefer_threads == 0
                                   ? current_socket_info[ALL_PROC]
                                   : std::min(current_socket_info[ALL_PROC], model_prefer_threads);
        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
        if (current_socket_info[ALL_PROC] == current_socket_info[MAIN_CORE_PROC]) {
            stream_info[PROC_TYPE] = MAIN_CORE_PROC;
            distribute_streams_per_node(MAIN_CORE_PROC, current_socket_info);
        } else if (current_socket_info[ALL_PROC] == current_socket_info[EFFICIENT_CORE_PROC]) {
            stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
            distribute_streams_per_node(EFFICIENT_CORE_PROC, current_socket_info);
        } else {
            stream_info[PROC_TYPE] = ALL_PROC;
            add_mixed_stream(current_socket_info,
                             proc_type_table,
                             n_threads_per_stream,
                             IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                             ALL_PROC);
        }
        set_ids(current_socket_info);
    }

    void handle_latency_mode() {
        n_streams = 1;
        stream_info[NUMBER_OF_STREAMS] = n_streams;

        for (auto& n : proc_socket_table) {
            if (n[ALL_PROC] > 0) {
                current_socket_id = n[PROC_SOCKET_ID];
                break;
            }
        }

        if (input_threads > 0) {
            handle_latency_with_explicit_threads();
        } else if (has_tensor_parallel_policy() || (proc_type_table.size() == 1)) {
            handle_latency_tensor_parallel_or_single_socket();
        } else {
            handle_latency_multi_socket();
        }
    }

    void handle_throughput_explicit_streams() {
        n_streams = input_infer_requests > 0 ? std::min(input_infer_requests, input_streams) : input_streams;
        if (n_streams >= n_threads) {
            n_streams = n_threads;
            n_threads_per_stream = 1;
        } else {
            n_threads_per_stream =
                std::min((n_threads / n_streams),
                         proc_type_table[0][MAIN_CORE_PROC] == 0 ? proc_type_table[0][EFFICIENT_CORE_PROC]
                                                                 : proc_type_table[0][MAIN_CORE_PROC]);
            adjust_threads_per_stream();
        }
    }

    void handle_throughput_auto_no_preference() {
        int base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
        int n_proc = 0;

        if (proc_type_table.size() == 1) {
            n_proc = std::min(n_threads, proc_type_table[0][base_type]);
        } else {
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                n_proc = std::max(n_proc, proc_type_table[i][base_type]);
            }
            n_proc = std::min(n_threads, n_proc);
        }

        if (0 == n_proc % 4) {
            n_threads_per_stream = 4;
        } else if (0 == n_proc % 5) {
            n_threads_per_stream = 5;
        } else if (0 == n_proc % 3) {
            n_threads_per_stream = 3;
        } else if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0 &&
                    proc_type_table[0][EFFICIENT_CORE_PROC] != proc_type_table[0][ALL_PROC]) ||
                   (proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0)) {
            n_threads_per_stream = n_proc;
        } else {
            n_threads_per_stream = (n_proc > 16) ? 4 : std::max(1, (n_proc / 4));
        }

        if (input_threads > 0) {
            n_streams = n_threads / n_threads_per_stream;
        } else {
            n_streams = 0;
            for (size_t i = MAIN_CORE_PROC; i <= HYPER_THREADING_PROC; i++) {
                n_streams += proc_type_table[0][i] / n_threads_per_stream;
            }
        }

        if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
            n_streams = input_infer_requests;
            if (proc_type_table.size() == 1) {
                n_threads_per_stream = std::min((n_threads / n_streams), n_proc);
            } else {
                n_threads_per_stream = (n_threads / n_streams);
            }
        } else {
            while ((n_streams * 2 <= n_threads_per_stream) && (n_threads_per_stream > 1)) {
                n_threads_per_stream = (n_threads_per_stream / 2);
                n_streams = (n_threads / n_threads_per_stream);
            }
        }
    }

    void handle_throughput_hybrid_single_thread_prefer() {
        n_streams = (n_threads >= proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC] +
                                      proc_type_table[0][LP_EFFICIENT_CORE_PROC])
                        ? (n_threads - proc_type_table[0][EFFICIENT_CORE_PROC] / 2 -
                           proc_type_table[0][LP_EFFICIENT_CORE_PROC] / 2)
                        : static_cast<int>(proc_type_table[0][MAIN_CORE_PROC] +
                                           (n_threads - proc_type_table[0][MAIN_CORE_PROC]) / 2);
        n_streams = input_infer_requests > 0 ? std::min(n_streams, input_infer_requests) : n_streams;
        n_threads_per_stream = -1;
    }

    void handle_throughput_general_model_prefer() {
        int model_threads = [&]() {
            if (n_threads == 1) {
                return 1;
            }
            if (model_prefer_threads > n_threads) {
                return n_threads / 2;
            }
            return model_prefer_threads;
        }();
        n_streams = ((n_threads + model_threads - 1) / model_threads);
        if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
            n_streams = input_infer_requests;
            n_threads_per_stream = (n_threads / n_streams);
            adjust_threads_per_stream();
        } else {
            n_threads_per_stream = model_threads > 0 ? model_threads : (n_threads / n_streams);
        }
    }

    void handle_throughput_mode() {
        n_threads =
            input_threads > 0 ? std::min(proc_type_table[0][ALL_PROC], input_threads) : proc_type_table[0][ALL_PROC];

        if ((input_streams_changed) && (input_streams > 0)) {
            handle_throughput_explicit_streams();
        } else if (0 == model_prefer_threads) {
            handle_throughput_auto_no_preference();
        } else if ((1 == model_prefer_threads) &&
                   ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) ||
                    (proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0)) &&
                   (proc_type_table[0][MAIN_CORE_PROC] > 0) && (n_threads > proc_type_table[0][MAIN_CORE_PROC])) {
            handle_throughput_hybrid_single_thread_prefer();
        } else {
            handle_throughput_general_model_prefer();
        }
    }

    void populate_table_tensor_parallel() {
        for (auto& row : proc_socket_table) {
            stream_info[THREADS_PER_STREAM] = std::min(TP_CPU_LIMIT, n_threads_per_stream);
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                if ((proc_type_table[i][PROC_SOCKET_ID] == row[PROC_SOCKET_ID]) &&
                    (proc_type_table[i][MAIN_CORE_PROC] >= stream_info[THREADS_PER_STREAM])) {
                    add_single_stream(proc_type_table[i],
                                      {proc_type_table[i]},
                                      stream_info[THREADS_PER_STREAM],
                                      IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_FOR_SOCKET);
                    break;
                }
            }
            if (stream_info[STREAM_SOCKET_ID] == row[PROC_SOCKET_ID]) {
                continue;
            }
            stream_info[THREADS_PER_STREAM] = std::min(stream_info[THREADS_PER_STREAM], row[ALL_PROC]);
            add_single_stream(row,
                              proc_type_table,
                              stream_info[THREADS_PER_STREAM],
                              IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_FOR_SOCKET);
        }

        stream_info = streams_info_table[0];
        stream_info[NUMBER_OF_STREAMS] = 1;
        for (size_t n = 1; n < streams_info_table.size(); n++) {
            if (streams_info_table[n][NUMBER_OF_STREAMS] == -1) {
                if (stream_info[PROC_TYPE] != streams_info_table[n][PROC_TYPE]) {
                    stream_info[PROC_TYPE] = ALL_PROC;
                }
                stream_info[THREADS_PER_STREAM] += streams_info_table[n][THREADS_PER_STREAM];
                if (stream_info[STREAM_NUMA_NODE_ID] != streams_info_table[n][STREAM_NUMA_NODE_ID]) {
                    stream_info[STREAM_NUMA_NODE_ID] = -1;
                }
                if (stream_info[STREAM_SOCKET_ID] != streams_info_table[n][STREAM_SOCKET_ID]) {
                    stream_info[STREAM_SOCKET_ID] = -1;
                }
            }
        }
        streams_info_table.insert(streams_info_table.begin(), stream_info);
    }

    static void subtract_allocated_from(std::vector<std::vector<int>>& remain,
                                        const std::vector<std::vector<int>>& table,
                                        size_t begin,
                                        size_t end) {
        for (size_t i = begin; i < end; i++) {
            if ((table[i][STREAM_NUMA_NODE_ID] < 0) || (table[i][STREAM_SOCKET_ID] < 0)) {
                continue;
            }
            for (auto& row : remain) {
                if ((table[i][STREAM_NUMA_NODE_ID] == row[PROC_NUMA_NODE_ID]) &&
                    (table[i][STREAM_SOCKET_ID] == row[PROC_SOCKET_ID])) {
                    row[table[i][PROC_TYPE]] -= (table[i][NUMBER_OF_STREAMS] == 0 ? 1 : table[i][NUMBER_OF_STREAMS]) *
                                                table[i][THREADS_PER_STREAM];
                }
            }
        }
    }

    void populate_table_normal(int total_streams) {
        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;

        // Pass 1: allocate streams to homogeneous core types per node.
        for (int n_type = MAIN_CORE_PROC; (n_type <= HYPER_THREADING_PROC) && (n_streams > 0); n_type++) {
            if (proc_type_table.size() == 1) {
                if (proc_type_table[0][n_type] >= stream_info[THREADS_PER_STREAM]) {
                    distribute_streams_per_node(n_type, proc_type_table[0]);
                }
            } else {
                for (size_t n_node = 1; (n_node < proc_type_table.size()) && (n_streams > 0); n_node++) {
                    if ((proc_type_table[n_node][n_type] >= stream_info[THREADS_PER_STREAM]) &&
                        ((current_socket_id < 0) || (proc_type_table[n_node][PROC_SOCKET_ID] == current_socket_id))) {
                        distribute_streams_per_node(n_type, proc_type_table[n_node]);
                    }
                }
            }
        }

        // Pass 2: if nothing was allocated yet, try mixed streams.
        if (total_streams == n_streams) {
            if (proc_type_table.size() == 1) {
                if (proc_type_table[0][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                    add_mixed_stream(proc_type_table[0],
                                     proc_type_table,
                                     n_threads_per_stream,
                                     IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                                     ALL_PROC);
                    n_streams--;
                }
            } else {
                for (size_t n_node = 0; (n_node < proc_socket_table.size()) && (n_streams > 0); n_node++) {
                    if ((proc_socket_table[n_node][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) &&
                        ((current_socket_id < 0) || (proc_socket_table[n_node][PROC_SOCKET_ID] == current_socket_id))) {
                        add_mixed_stream(proc_socket_table[n_node],
                                         proc_type_table,
                                         n_threads_per_stream,
                                         IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                                         ALL_PROC);
                        n_streams--;
                    }
                }
            }
        }

        // Pass 3: overflow — pack remaining streams into whatever capacity is left.
        if (n_streams > 0) {
            std::vector<std::vector<int>> remain_proc_type_table(proc_type_table);
            size_t stream_table_size = streams_info_table.size();

            subtract_allocated_from(remain_proc_type_table, streams_info_table, 0, stream_table_size);

            while (n_streams > 0) {
                add_mixed_stream(proc_type_table[0],
                                 remain_proc_type_table,
                                 n_threads_per_stream,
                                 IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                                 ALL_PROC);

                if (stream_table_size == streams_info_table.size()) {
                    break;
                }
                n_streams--;
                int numa_node_id = streams_info_table[stream_table_size + 1][STREAM_NUMA_NODE_ID];
                int socket_id = streams_info_table[stream_table_size + 1][STREAM_SOCKET_ID];
                for (size_t i = stream_table_size + 1; i < streams_info_table.size(); i++) {
                    numa_node_id = numa_node_id == streams_info_table[i][STREAM_NUMA_NODE_ID] ? numa_node_id : -1;
                    socket_id = socket_id == streams_info_table[i][STREAM_SOCKET_ID] ? socket_id : -1;
                    for (auto& row : remain_proc_type_table) {
                        if ((streams_info_table[i][STREAM_NUMA_NODE_ID] == row[PROC_NUMA_NODE_ID]) &&
                            (streams_info_table[i][STREAM_SOCKET_ID] == row[PROC_SOCKET_ID])) {
                            row[streams_info_table[i][PROC_TYPE]] -= (streams_info_table[i][NUMBER_OF_STREAMS] == 0
                                                                          ? 1
                                                                          : streams_info_table[i][NUMBER_OF_STREAMS]) *
                                                                     streams_info_table[i][THREADS_PER_STREAM];
                        }
                    }
                }
                streams_info_table[stream_table_size][STREAM_NUMA_NODE_ID] = numa_node_id;
                streams_info_table[stream_table_size][STREAM_SOCKET_ID] = socket_id;
                stream_table_size = streams_info_table.size();
            }
        }

        if ((total_streams == 1) && (proc_type_table.size() == 1) && enable_tensor_parallel &&
            has_tensor_parallel_policy()) {
            streams_info_table.push_back(streams_info_table[0]);
            streams_info_table.push_back(streams_info_table[0]);
            streams_info_table[0][THREADS_PER_STREAM] = streams_info_table[0][THREADS_PER_STREAM] * 2;
            streams_info_table[1][NUMBER_OF_STREAMS] = -1;
            streams_info_table[2][NUMBER_OF_STREAMS] = -1;
        }
    }

    /**
     * Handle the case where latency mode already pre-set stream_info[PROC_TYPE]
     * and we have a single-socket configuration.
     */
    void populate_table_single_socket_preset() {
        if (stream_info[PROC_TYPE] == ALL_PROC) {
            add_mixed_stream(proc_socket_table[0],
                             proc_type_table,
                             n_threads_per_stream,
                             IStreamsExecutor::Config::StreamsMode::SUB_STREAMS_NULL,
                             ALL_PROC);
        } else if (stream_info[PROC_TYPE] == MAIN_CORE_PROC) {
            if (stream_info[THREADS_PER_STREAM] == proc_socket_table[0][MAIN_CORE_PROC]) {
                streams_info_table.push_back(stream_info);
            } else {
                stream_info[PROC_TYPE] = ALL_PROC;
                streams_info_table.push_back(stream_info);
                stream_info[NUMBER_OF_STREAMS] = 0;
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] = proc_socket_table[0][MAIN_CORE_PROC];
                streams_info_table.push_back(stream_info);
                stream_info[PROC_TYPE] = HYPER_THREADING_PROC;
                stream_info[THREADS_PER_STREAM] = proc_socket_table[0][HYPER_THREADING_PROC];
                streams_info_table.push_back(stream_info);
            }
        } else {
            streams_info_table.push_back(stream_info);
        }
    }

    /**
     * Phase 3: populate the streams_info_table based on the n_streams,
     * n_threads_per_stream, and (optionally pre-set) stream_info[PROC_TYPE]
     * computed in earlier phases.
     */
    void populate_streams_info_table() {
        int total_streams = n_streams;

        if (stream_info[PROC_TYPE] == INIT_VAL) {
            bool is_multi_socket_tp = (n_streams == 1) && (proc_type_table.size() > 1) && has_tensor_parallel_policy();

            if (is_multi_socket_tp) {
                populate_table_tensor_parallel();
            } else {
                populate_table_normal(total_streams);
            }
        } else if (proc_type_table.size() == 1) {
            populate_table_single_socket_preset();
        }
    }

    // ── top-level entry point ───────────────────────────────────────────

    std::vector<std::vector<int>> build() {
        stream_info.assign(CPU_STREAMS_TABLE_SIZE, INIT_VAL);

        build_socket_table();

        if (is_latency_mode()) {
            handle_latency_mode();
        } else {
            handle_throughput_mode();
        }

        populate_streams_info_table();

        return streams_info_table;
    }

private:
    /**
     * Check whether a NUMA-node row's socket matches the current socket-
     * selection mode used by add_mixed_stream.
     *
     * @param n_mode      3 = same as current_socket_id,
     *                    2 = different from current_socket_id,
     *                    1 = any socket.
     * @param socket_id   The stream's target socket (-1 means any).
     * @param row_socket  The socket id of the row being tested.
     */
    [[nodiscard]] bool matches_socket_mode(int n_mode, int socket_id, int row_socket) const {
        switch (n_mode) {
        case 3:
            return (current_socket_id == row_socket) && ((socket_id < 0) || (socket_id == row_socket));
        case 2:
            return (current_socket_id != row_socket) && ((socket_id < 0) || (socket_id == row_socket));
        case 1:
            return (socket_id < 0) || (socket_id == row_socket);
        default:
            return false;
        }
    }
};

std::vector<std::vector<int>> get_streams_info_table(
    int input_streams,
    bool input_streams_changed,
    int input_threads,
    int input_infer_requests,
    int model_prefer_threads,
    bool enable_tensor_parallel,
    const std::string& input_perf_hint,
    const std::set<ov::hint::ModelDistributionPolicy>& hint_model_distribution_policy,
    const std::vector<std::vector<int>>& proc_type_table) {
    StreamsInfoBuilder builder{input_streams,
                               input_streams_changed,
                               input_threads,
                               input_infer_requests,
                               model_prefer_threads,
                               enable_tensor_parallel,
                               input_perf_hint,
                               hint_model_distribution_policy,
                               proc_type_table};
    return builder.build();
}

std::vector<std::vector<int>> get_streams_rank_table(const std::vector<std::vector<int>>& streams_info_table,
                                                     const int input_rank_level,
                                                     int& num_sub_streams) {
    std::vector<std::vector<int>> rank_table = {};
    num_sub_streams = 0;
    std::vector<int> init_rank = {};
    int rank_level = input_rank_level == 0 ? 1 : input_rank_level;
    init_rank.resize(rank_level, 0);

    for (const auto& row : streams_info_table) {
        if (row[NUMBER_OF_STREAMS] < 0) {
            for (int i = 0; i < abs(row[NUMBER_OF_STREAMS]); i++) {
                init_rank[rank_level - 1] = num_sub_streams + i;
                rank_table.push_back(init_rank);
            }
            num_sub_streams -= row[NUMBER_OF_STREAMS];
        }
    }
    if (rank_level == 2) {
        for (int i = num_sub_streams / 2; i < num_sub_streams; i++) {
            rank_table[i][0] = 1;
            rank_table[i][1] -= num_sub_streams / 2;
        }
    }
    return rank_table;
}

#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__)
static void configure_arm64_linux_threads(Config& config,
                                          const std::vector<std::vector<int>>& proc_type_table,
                                          bool int8_intensive,
                                          bool is_LLM) {
    using namespace ThreadPreferenceConstants;
    config.modelPreferThreadsThroughput = ARM64_THREADS_DEFAULT;
    if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::cpu_isa_t::sve_128)) {
        config.modelPreferThreadsThroughput = ARM64_THREADS_SVE;
    }

    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

    bool use_all_cores = should_use_all_cores_for_latency(main_cores, efficient_cores, int8_intensive);

    if (use_all_cores && (!is_LLM || should_use_ecores_for_llm(efficient_cores, main_cores))) {
        config.modelPreferThreadsLatency = main_cores + efficient_cores;
    } else {
        config.modelPreferThreadsLatency = main_cores;
    }
}
#endif

#if defined(OPENVINO_ARCH_ARM) && defined(__linux__)
void configure_arm_linux_threads(Config& config,
                                 const std::vector<std::vector<int>>& proc_type_table,
                                 const ov::MemBandwidthPressure& tolerance,
                                 bool int8_intensive,
                                 bool is_LLM) {
    using namespace ThreadPreferenceConstants;
    config.modelPreferThreadsThroughput = ARM_THREADS_DEFAULT;

    if (tolerance.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
        if (tolerance.ratio_compute_convs == ov::MemBandwidthPressure::ALL) {
            config.modelPreferThreadsThroughput = ARM_THREADS_HIGH;
        }
    } else if ((tolerance.max_mem_tolerance < ov::MemBandwidthPressure::LIMITED) &&
               ((tolerance.ratio_mem_limited_deconvs > ov::MemBandwidthPressure::LIMITED) ||
                (tolerance.ratio_mem_limited_gemms > ov::MemBandwidthPressure::LIMITED))) {
        config.modelPreferThreadsThroughput = ARM_THREADS_HIGH;
    }

    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

    bool use_all_cores = should_use_all_cores_for_latency(main_cores, efficient_cores, int8_intensive);

    if (use_all_cores && (!is_LLM || should_use_ecores_for_llm(efficient_cores, main_cores))) {
        config.modelPreferThreadsLatency = main_cores + efficient_cores;
    } else {
        config.modelPreferThreadsLatency = main_cores;
    }
}
#endif

#if (defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__APPLE__)
void configure_apple_threads(Config& config,
                             const std::vector<std::vector<int>>& proc_type_table,
                             const ov::MemBandwidthPressure& tolerance,
                             float memThresholdAssumeLimitedForISA,
                             bool int8_intensive,
                             bool is_LLM) {
    using namespace ThreadPreferenceConstants;
    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

    if ((proc_type_table.size() == 1) && (efficient_cores > 0)) {
        config.modelPreferThreadsLatency = main_cores > efficient_cores ? main_cores : proc_type_table[0][ALL_PROC];
    } else {
        bool use_all_cores = should_use_all_cores_for_latency(main_cores, efficient_cores, int8_intensive);

        if (use_all_cores && (!is_LLM || should_use_ecores_for_llm(efficient_cores, main_cores))) {
            config.modelPreferThreadsLatency = main_cores + efficient_cores;
        } else {
            config.modelPreferThreadsLatency = main_cores;
        }
    }

    config.modelPreferThreadsThroughput = APPLE_THREADS_MINIMAL;

    if (tolerance.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
        if (is_network_compute_limited(tolerance)) {
            config.modelPreferThreadsThroughput = APPLE_THREADS_HIGH;
        }
    } else if (is_below_isa_threshold(tolerance.max_mem_tolerance, memThresholdAssumeLimitedForISA)) {
        config.modelPreferThreadsThroughput = APPLE_THREADS_MINIMAL;
    } else if (is_below_general_threshold(tolerance.max_mem_tolerance)) {
        config.modelPreferThreadsThroughput = APPLE_THREADS_MINIMAL;
    } else if (tolerance.ratio_mem_limited_deconvs > ov::MemBandwidthPressure::LIMITED &&
               tolerance.ratio_compute_convs < ov::MemBandwidthPressure::ALL) {
        config.modelPreferThreadsThroughput = APPLE_THREADS_HIGH;
    } else if (tolerance.ratio_mem_limited_deconvs <= ov::MemBandwidthPressure::LIMITED &&
               tolerance.ratio_mem_limited_convs <= ov::MemBandwidthPressure::LIMITED &&
               tolerance.ratio_compute_convs > ov::MemBandwidthPressure::LIMITED) {
        config.modelPreferThreadsThroughput = APPLE_THREADS_LOW;
    }
}
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_RISCV64)
void configure_x86_hybrid_threads(Config& config,
                                  const std::vector<std::vector<int>>& proc_type_table,
                                  const ov::MemBandwidthPressure& tolerance,
                                  bool int8_intensive,
                                  bool is_LLM) {
    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];
    const int lp_efficient_cores = proc_type_table[0][LP_EFFICIENT_CORE_PROC];

    const bool is_hybrid_applicable = (main_cores < config.threads || config.threads == 0) &&
                                      (ov::get_number_of_blocked_cores() != 0 || lp_efficient_cores > 0) &&
                                      efficient_cores <= 2 * main_cores;

    if (is_hybrid_applicable) {
        if (is_LLM) {
            config.modelPreferThreadsLatency = main_cores;
        } else {
            config.modelPreferThreadsLatency = main_cores + efficient_cores;
            determine_tbb_partitioner_and_threads(config, proc_type_table, tolerance, int8_intensive);
        }
    } else {
        // Fall back to default latency preference logic
        bool use_all_cores = should_use_all_cores_for_latency(main_cores, efficient_cores, int8_intensive);

        if (use_all_cores && (!is_LLM || should_use_ecores_for_llm(efficient_cores, main_cores))) {
            config.modelPreferThreadsLatency = main_cores + efficient_cores;
        } else {
            config.modelPreferThreadsLatency = main_cores;
        }
    }
}

void configure_x86_hybrid_lp_threads(Config& config,
                                     const std::vector<std::vector<int>>& proc_type_table,
                                     const ov::MemBandwidthPressure& tolerance) {
    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int lp_efficient_cores = proc_type_table[0][LP_EFFICIENT_CORE_PROC];

    if (is_lp_auto_case_1(tolerance) || is_lp_auto_case_2(tolerance) || is_lp_auto_case_3(tolerance) ||
        is_lp_auto_case_4(tolerance) || is_lp_auto_case_5(tolerance)) {
        config.modelPreferThreadsLatency = main_cores + lp_efficient_cores;
        config.tbbPartitioner =
            config.tbbPartitioner == TbbPartitioner::NONE ? TbbPartitioner::AUTO : config.tbbPartitioner;
    } else {
        if (is_lp_main_core_case_1(tolerance) || is_lp_main_core_case_2(tolerance)) {
            config.modelPreferThreadsLatency = main_cores;
        } else {
            config.modelPreferThreadsLatency = main_cores + lp_efficient_cores;
        }
        config.tbbPartitioner =
            config.tbbPartitioner == TbbPartitioner::NONE ? TbbPartitioner::STATIC : config.tbbPartitioner;
    }
}

void configure_x86_non_hybrid_threads(Config& config, const std::vector<std::vector<int>>& proc_type_table) {
    const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
    const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

    config.modelPreferThreadsLatency = main_cores > 0 ? main_cores : efficient_cores;
}

void configure_x86_throughput_threads(Config& config,
                                      const std::vector<std::vector<int>>& proc_type_table,
                                      const ov::MemBandwidthPressure& tolerance,
                                      float memThresholdAssumeLimitedForISA) {
    config.modelPreferThreadsThroughput = 0;

    if (tolerance.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
        if (is_network_compute_limited(tolerance)) {
            config.modelPreferThreadsThroughput = 1;
        }
    } else if (is_below_isa_threshold(tolerance.max_mem_tolerance, memThresholdAssumeLimitedForISA)) {
        config.modelPreferThreadsThroughput = 1;
    } else if (is_below_general_threshold(tolerance.max_mem_tolerance)) {
        config.modelPreferThreadsThroughput = 2;
    }

    // Adjust for hyperthreading on non-hybrid systems
    if (config.modelPreferThreadsThroughput == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 &&
        (proc_type_table[0][HYPER_THREADING_PROC] == proc_type_table[0][MAIN_CORE_PROC])) {
        config.modelPreferThreadsThroughput = 2;
    }
}
#endif

int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>>& proc_type_table,
                             const std::shared_ptr<ov::Model>& model,
                             Config& config,
                             int num_sockets,
                             float isaSpecificThreshold) {
    if (num_sockets == -1) {
        num_sockets = get_num_sockets();
    }

    if (config.modelPreferThreads == -1) {
        const bool int8_intensive = ov::op::util::has_op_with_type<ov::op::v0::FakeQuantize>(model);
        const bool is_LLM = config.modelType != Config::ModelType::CNN;

        config.modelPreferThreads = 0;

#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__)
        configure_arm64_linux_threads(config, proc_type_table, int8_intensive, is_LLM);
        (void)isaSpecificThreshold;
#else

        if (isaSpecificThreshold == -1) {
#    if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
            isaSpecificThreshold = get_isa_threshold_multiplier(dnnl::get_effective_cpu_isa());
#    else
            isaSpecificThreshold = 1.0F;
#    endif
        }

        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED / isaSpecificThreshold;
#    if defined(OPENVINO_ARCH_RISCV64)
        // oneDNN C++ API (dnnl::utils::get_cache_size) is not available on RISC-V;
        // use a fallback value for L2 cache size.
        const float L2_cache_size = 1.0F;
#    else
        const float L2_cache_size = static_cast<float>(dnnl::utils::get_cache_size(2 /*level*/, true /*per core */));
#    endif
        ov::MemBandwidthPressure networkToleranceForLowCache =
            ov::mem_bandwidth_pressure_tolerance(model,
                                                 L2_cache_size,
                                                 memThresholdAssumeLimitedForISA,
                                                 config.inferencePrecision);

#    if defined(OPENVINO_ARCH_ARM) && defined(__linux__)
        configure_arm_linux_threads(config, proc_type_table, networkToleranceForLowCache, int8_intensive, is_LLM);

#    elif (defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__APPLE__)
        configure_apple_threads(config,
                                proc_type_table,
                                networkToleranceForLowCache,
                                memThresholdAssumeLimitedForISA,
                                int8_intensive,
                                is_LLM);

#    else
#        if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_RISCV64)
        const int main_cores = proc_type_table[0][MAIN_CORE_PROC];
        const int efficient_cores = proc_type_table[0][EFFICIENT_CORE_PROC];
        const int lp_efficient_cores = proc_type_table[0][LP_EFFICIENT_CORE_PROC];

        if (efficient_cores > 0 && main_cores > 0) {
            configure_x86_hybrid_threads(config, proc_type_table, networkToleranceForLowCache, int8_intensive, is_LLM);
        } else if (efficient_cores == 0 && main_cores * 2 <= lp_efficient_cores) {
            configure_x86_hybrid_lp_threads(config, proc_type_table, networkToleranceForLowCache);
        } else {
            configure_x86_non_hybrid_threads(config, proc_type_table);
        }
#        endif

        configure_x86_throughput_threads(config,
                                         proc_type_table,
                                         networkToleranceForLowCache,
                                         memThresholdAssumeLimitedForISA);
#    endif
#endif
    }

    if (num_streams > num_sockets || num_streams == 0) {
        config.modelPreferThreads = config.modelPreferThreadsThroughput;
    } else {
        config.modelPreferThreads = config.modelPreferThreadsLatency;
    }

    return config.modelPreferThreads;
}

std::vector<std::vector<int>> generate_stream_info(const int streams,
                                                   const int input_numa_node_id,
                                                   const std::shared_ptr<ov::Model>& model,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream) {
    OPENVINO_ASSERT(!proc_type_table.empty() && proc_type_table[0][ALL_PROC] != 0,
                    "proc_type_table is empty. No CPU resources available!");
    int model_prefer_threads = preferred_nthreads_per_stream;
    proc_type_table = apply_scheduling_core_type(config.schedulingCoreType, proc_type_table);

    proc_type_table = apply_hyper_threading(config.enableHyperThreading,
                                            config.changedHyperThreading,
                                            ov::util::to_string(config.hintPerfMode),
                                            proc_type_table);
    if (-1 == preferred_nthreads_per_stream) {
        model_prefer_threads = get_model_prefer_threads(streams, proc_type_table, model, config);
    }

    if (proc_type_table.size() > 1) {
        int cur_numa_node_id = input_numa_node_id < 0 ? get_current_numa_node_id() : input_numa_node_id;
        sort_table_by_numa_node_id(cur_numa_node_id, proc_type_table);
    }
    OPENVINO_ASSERT(!proc_type_table.empty() && proc_type_table[0][ALL_PROC] != 0,
                    "proc_type_table is empty. No valid CPU resources available!");
    auto streams_info_table = get_streams_info_table(config.streams,
                                                     config.streamsChanged,
                                                     config.threads,
                                                     config.hintNumRequests,
                                                     model_prefer_threads,
                                                     config.enableTensorParallel,
                                                     ov::util::to_string(config.hintPerfMode),
                                                     config.modelDistributionPolicy,
                                                     proc_type_table);
    config.tbbPartitioner =
        config.tbbPartitioner == TbbPartitioner::NONE ? TbbPartitioner::STATIC : config.tbbPartitioner;
    OPENVINO_ASSERT(!streams_info_table.empty(), "streams_info_table is empty!");
    if (config.modelDistributionPolicy.find(ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL) !=
        config.modelDistributionPolicy.end()) {
        config.streamsRankTable =
            get_streams_rank_table(streams_info_table, config.streamsRankLevel, config.numSubStreams);
    }

    config.enableCpuPinning = check_cpu_pinning(config.enableCpuPinning,
                                                config.changedCpuPinning,
                                                config.enableCpuReservation,
                                                streams_info_table);

    config.streamExecutorConfig = IStreamsExecutor::Config{"CPUStreamsExecutor",
                                                           config.streams,
                                                           config.threadsPerStream,
                                                           ov::hint::SchedulingCoreType::ANY_CORE,
                                                           config.enableCpuReservation,
                                                           config.enableCpuPinning,
                                                           true,
                                                           std::move(streams_info_table),
                                                           {},
                                                           false};
    return proc_type_table;
}

void get_num_streams(const int streams, const std::shared_ptr<ov::Model>& model, Config& config) {
    {
        std::lock_guard<std::mutex> lock{_streams_executor_mutex};
        std::vector<std::vector<int>> proc_type_table = get_proc_type_table();

        generate_stream_info(streams, -1, model, config, proc_type_table);
    }
}

}  // namespace ov::intel_cpu
