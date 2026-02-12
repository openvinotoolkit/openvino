// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cpu_streams_calculation_prefer_threads.hpp
 * @brief Helper functions and constants for get_model_prefer_threads implementation.
 *
 * This header provides:
 * - Named constants replacing magic numbers for better maintainability
 * - Helper functions for thread preference calculations
 * - Platform-specific configuration utilities
 * - TBB partitioner decision logic
 */

#pragma once

#include <vector>

#include "config.h"
#include "onednn/dnnl.h"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

namespace ov::intel_cpu {
namespace ThreadPreferenceConstants {

constexpr int INT8_EFFICIENCY_THRESHOLD = 4;
constexpr int FP32_EFFICIENCY_THRESHOLD = 2;

constexpr float ISA_THRESHOLD_SSE41 = 0.5F;
constexpr float ISA_THRESHOLD_AVX2 = 1.0F;
constexpr float ISA_THRESHOLD_VNNI = 2.0F;
constexpr float ISA_THRESHOLD_AMX = 4.0F;

constexpr float MEM_TOLERANCE_VERY_HIGH = 50.0F;
constexpr float MEM_TOLERANCE_HIGH = 4.5F;
constexpr float MEM_TOLERANCE_MEDIUM = 2.5F;
constexpr float MEM_TOLERANCE_MEDIUM_LOW = 0.5F;
constexpr float MEM_TOLERANCE_LOW = 0.2F;
constexpr float MEM_TOLERANCE_SECONDARY_LOW = 0.08F;
constexpr float MEM_TOLERANCE_VERY_LOW = 0.06F;

constexpr float CONV_RATIO_VERY_HIGH = 0.9F;
constexpr float CONV_RATIO_HIGH = 0.8F;
constexpr float CONV_RATIO_MEDIUM = 0.6F;
constexpr float CONV_RATIO_MEDIUM_LOW = 0.5F;
constexpr float CONV_RATIO_LOW = 0.46F;
constexpr float CONV_RATIO_MINIMAL = 0.28F;
constexpr float CONV_RATIO_VERY_LOW = 0.2F;
constexpr float CONV_RATIO_ULTRA_LOW = 0.1F;

constexpr float GEMM_RATIO_HIGH = 0.14F;
constexpr float GEMM_RATIO_LOW = 0.05F;

constexpr int ECORE_RATIO_THRESHOLD = 2;

constexpr int ARM64_THREADS_DEFAULT = 8;
constexpr int ARM64_THREADS_SVE = 16;
constexpr int ARM_THREADS_DEFAULT = 4;
constexpr int ARM_THREADS_HIGH = 8;

constexpr int APPLE_THREADS_MINIMAL = 1;
constexpr int APPLE_THREADS_LOW = 2;
constexpr int APPLE_THREADS_HIGH = 4;

}  // namespace ThreadPreferenceConstants

/**
 * @brief Get ISA-specific threshold multiplier for memory bandwidth calculations
 *
 * @param isa The detected CPU ISA capability
 * @return Threshold multiplier based on ISA compute capability
 */
inline float get_isa_threshold_multiplier(dnnl::cpu_isa isa) {
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

/**
 * @brief Determine if all cores (P-cores + E-cores) should be used for latency mode
 *
 * This function implements the heuristic that all cores should be used when
 * the number of P-cores is small relative to E-cores, adjusted by workload type.
 *
 * @param main_cores Number of performance (main/big) cores
 * @param efficient_cores Number of efficiency (little) cores
 * @param int8_intensive Whether the workload is INT8-intensive (uses VNNI/DP4A)
 * @return true if all cores should be utilized for latency mode
 */
inline bool should_use_all_cores_for_latency(int main_cores, int efficient_cores, bool int8_intensive) {
    using namespace ThreadPreferenceConstants;
    const int threshold = int8_intensive ? INT8_EFFICIENCY_THRESHOLD : FP32_EFFICIENCY_THRESHOLD;
    return main_cores * threshold <= efficient_cores;
}

/**
 * @brief Determine if E-cores should be used for LLM workloads
 *
 * For LLM workloads, E-cores are only beneficial when there are significantly
 * more E-cores than P-cores (more than 2x).
 *
 * @param efficient_cores Number of efficiency cores
 * @param main_cores Number of performance cores
 * @return true if E-cores should be included for LLM workloads
 */
inline bool should_use_ecores_for_llm(int efficient_cores, int main_cores) {
    using namespace ThreadPreferenceConstants;
    return efficient_cores > ECORE_RATIO_THRESHOLD * main_cores;
}

/**
 * @brief Check main core case 1: High ratio of memory-limited convolutions
 *
 * When >80% of convolutions are memory-limited, prefer main cores only
 * to avoid memory bandwidth contention.
 */
inline bool is_main_core_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs > CONV_RATIO_HIGH;
}

/**
 * @brief Check main core case 2: No convolutions but high memory tolerance
 *
 * Network has no convolutions but high memory tolerance (compute-bound),
 * prefer main cores for better single-thread performance.
 */
inline bool is_main_core_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs == 0.0F && tolerance.ratio_compute_convs == 0.0F &&
           tolerance.max_mem_tolerance >= MEM_TOLERANCE_HIGH;
}

/**
 * @brief Check main core case 3: Mostly light convolutions with partial compute
 *
 * Network has no memory-limited convs, some compute convs, and >90% light convs.
 */
inline bool is_main_core_case_3(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs == 0.0F && tolerance.ratio_compute_convs > 0.0F &&
           tolerance.ratio_compute_convs < 1.0F &&
           static_cast<float>(tolerance.total_light_convs) >
               CONV_RATIO_VERY_HIGH * static_cast<float>(tolerance.total_convs);
}

/**
 * @brief Check main core case 4: Mixed workload with significant light convolutions
 *
 * Network has both memory-limited and compute convs, with >46% light convs.
 */
inline bool is_main_core_case_4(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_mem_limited_convs > 0.0F && tolerance.ratio_compute_convs > 0.0F &&
           static_cast<float>(tolerance.total_light_convs) > CONV_RATIO_LOW * static_cast<float>(tolerance.total_convs);
}

/**
 * @brief Check static partitioner case 1: No recognized nodes
 *
 * When network has no recognized nodes, use static partitioner for predictability.
 */
inline bool is_static_partitioner_case_1(const ov::MemBandwidthPressure& tolerance) {
    return tolerance.total_nodes == 0;
}

/**
 * @brief Check static partitioner case 2: Majority light convolutions
 *
 * When >60% of convolutions are light, static partitioner provides better load balance.
 */
inline bool is_static_partitioner_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 && static_cast<float>(tolerance.total_light_convs) >
                                            CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs);
}

/**
 * @brief Check static partitioner case 3 for systems with LP E-cores
 *
 * Complex condition for systems with low-power efficiency cores.
 */
inline bool is_static_partitioner_case_3_with_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
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

/**
 * @brief Check static partitioner case 3 for systems without LP E-cores
 */
inline bool is_static_partitioner_case_3_without_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 &&
           static_cast<float>(tolerance.total_light_convs) <=
               CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs) &&
           tolerance.ratio_compute_convs + tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_HIGH &&
           tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_LOW && tolerance.ratio_mem_limited_gemms == 0.0F &&
           tolerance.ratio_mem_limited_adds < CONV_RATIO_MINIMAL &&
           tolerance.max_mem_tolerance >= MEM_TOLERANCE_VERY_LOW;
}

/**
 * @brief Check static partitioner case 4 for systems with LP E-cores
 *
 * No convolutions, but either high memory tolerance or significant GEMM ratio.
 */
inline bool is_static_partitioner_case_4_with_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 &&
           (tolerance.max_mem_tolerance > MEM_TOLERANCE_MEDIUM ||
            static_cast<float>(tolerance.total_gemms) >= GEMM_RATIO_HIGH * static_cast<float>(tolerance.total_nodes));
}

/**
 * @brief Check static partitioner case 4 for systems without LP E-cores
 *
 * No convolutions and low GEMM ratio.
 */
inline bool is_static_partitioner_case_4_without_lp_ecores(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 &&
           static_cast<float>(tolerance.total_gemms) < GEMM_RATIO_LOW * static_cast<float>(tolerance.total_nodes);
}

/**
 * @brief Check static partitioner case 5 (only for systems with LP E-cores)
 *
 * Complex condition for compute-heavy workloads with specific characteristics.
 */
inline bool is_static_partitioner_case_5(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 &&
           static_cast<float>(tolerance.total_light_convs) <=
               CONV_RATIO_MEDIUM * static_cast<float>(tolerance.total_convs) &&
           tolerance.ratio_compute_convs >= CONV_RATIO_VERY_HIGH * tolerance.ratio_mem_limited_convs &&
           tolerance.ratio_compute_convs == 1.0F && tolerance.ratio_mem_limited_adds == 1.0F &&
           static_cast<float>(tolerance.total_heavy_convs) >
               CONV_RATIO_ULTRA_LOW * static_cast<float>(tolerance.total_nodes);
}

/**
 * @brief Determine TBB partitioner strategy for hybrid architectures
 *
 * This function analyzes the network characteristics and determines whether
 * to use static or auto TBB partitioner, and whether to limit to main cores only.
 *
 * @param[in,out] config Configuration to update
 * @param[in] proc_type_table Processor topology table
 * @param[in] tolerance Network memory bandwidth pressure analysis
 * @param[in] int8_intensive Whether workload is INT8-intensive
 */
inline void determine_tbb_partitioner_and_threads(Config& config,
                                                  const std::vector<std::vector<int>>& proc_type_table,
                                                  const ov::MemBandwidthPressure& tolerance,
                                                  bool int8_intensive) {
    if (config.tbbPartitioner != TbbPartitioner::NONE) {
        return;  // Already configured by user
    }

    const bool has_lp_ecores = proc_type_table[0][LP_EFFICIENT_CORE_PROC] > 0;

    // Check main core cases (prefer main cores only for specific workloads)
    if (has_lp_ecores && int8_intensive && tolerance.total_convs > 0) {
        if (is_main_core_case_1(tolerance) || is_main_core_case_2(tolerance) || is_main_core_case_3(tolerance) ||
            is_main_core_case_4(tolerance)) {
            config.modelPreferThreadsLatency = proc_type_table[0][MAIN_CORE_PROC];
            config.tbbPartitioner = TbbPartitioner::STATIC;
            return;
        }
    }

    // Check static partitioner cases
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

/**
 * @brief Check if network is compute-limited (all convolutions/deconvolutions are compute-bound)
 */
inline bool is_network_compute_limited(const ov::MemBandwidthPressure& tolerance) {
    return tolerance.ratio_compute_convs == ov::MemBandwidthPressure::ALL ||
           tolerance.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL;
}

/**
 * @brief Check if network is below ISA-specific memory threshold
 */
inline bool is_below_isa_threshold(float max_tolerance, float memThresholdAssumeLimitedForISA) {
    return max_tolerance > memThresholdAssumeLimitedForISA;
}

/**
 * @brief Check if network is below general memory threshold
 */
inline bool is_below_general_threshold(float max_tolerance) {
    return max_tolerance > ov::MemBandwidthPressure::LIMITED;
}

inline bool is_lp_main_core_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs == 0 && tolerance.max_mem_tolerance > MEM_TOLERANCE_VERY_HIGH &&
           static_cast<float>(tolerance.total_gemms) < GEMM_RATIO_LOW * static_cast<float>(tolerance.total_nodes);
}

inline bool is_lp_main_core_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 0 && tolerance.total_gemms == 1 &&
           tolerance.max_mem_tolerance<MEM_TOLERANCE_MEDIUM_LOW&& static_cast<float>(
               tolerance.total_light_convs)> CONV_RATIO_HIGH *
               static_cast<float>(tolerance.total_convs);
}

inline bool is_lp_auto_case_1(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.total_convs > 52 && tolerance.ratio_compute_convs > 0 && tolerance.ratio_mem_limited_convs > 0 &&
           tolerance.ratio_mem_limited_convs < CONV_RATIO_VERY_LOW;
}

inline bool is_lp_auto_case_2(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.max_mem_tolerance < MEM_TOLERANCE_SECONDARY_LOW &&
           tolerance.ratio_compute_convs < CONV_RATIO_HIGH && tolerance.ratio_mem_limited_convs < CONV_RATIO_MEDIUM_LOW;
}

inline bool is_lp_auto_case_3(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.ratio_compute_convs > 0 && tolerance.ratio_compute_convs < CONV_RATIO_ULTRA_LOW &&
           tolerance.ratio_mem_limited_convs >= CONV_RATIO_VERY_LOW;
}

inline bool is_lp_auto_case_4(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.max_mem_tolerance > MEM_TOLERANCE_LOW && tolerance.ratio_compute_convs > CONV_RATIO_MEDIUM_LOW &&
           tolerance.ratio_mem_limited_adds > 0 &&
           static_cast<float>(tolerance.total_adds) < CONV_RATIO_VERY_LOW * static_cast<float>(tolerance.total_nodes);
}

inline bool is_lp_auto_case_5(const ov::MemBandwidthPressure& tolerance) {
    using namespace ThreadPreferenceConstants;
    return tolerance.max_mem_tolerance <= MEM_TOLERANCE_SECONDARY_LOW && tolerance.total_light_convs > 10;
}

}  // namespace ov::intel_cpu
