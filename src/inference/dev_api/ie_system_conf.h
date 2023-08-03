// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file ie_system_conf.h
 */

#pragma once

#include <exception>
#include <vector>

#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

namespace InferenceEngine {

/**
 * @brief      Checks whether OpenMP environment variables are defined
 * @ingroup    ie_dev_api_system_conf
 *
 * @param[in]  includeOMPNumThreads  Indicates if the omp number threads is included
 * @return     `True` if any OpenMP environment variable is defined, `false` otherwise
 */
inline bool checkOpenMpEnvVars(bool includeOMPNumThreads = true) {
    return ov::check_open_mp_env_vars(includeOMPNumThreads);
}

/**
 * @brief      Returns available CPU NUMA nodes (on Linux, and Windows [only with TBB], single node is assumed on all
 * other OSes)
 * @ingroup    ie_dev_api_system_conf
 * @return     NUMA nodes
 */
inline std::vector<int> getAvailableNUMANodes() {
    return ov::get_available_numa_nodes();
}

/**
 * @brief      Returns available CPU cores types (on Linux, and Windows) and ONLY with TBB, single core type is assumed
 * otherwise
 * @ingroup    ie_dev_api_system_conf
 * @return     Vector of core types
 */
inline std::vector<int> getAvailableCoresTypes() {
    return ov::get_available_cores_types();
}

/**
 * @brief      Returns number of CPU physical cores on Linux/Windows (which is considered to be more performance
 * friendly for servers) (on other OSes it simply relies on the original parallel API of choice, which usually uses the
 * logical cores). call function with 'false' to get #phys cores of all types call function with 'true' to get #phys
 * 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  bigCoresOnly Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of physical CPU cores.
 */
inline int getNumberOfCPUCores(bool bigCoresOnly = false) {
    return ov::get_number_of_cpu_cores(bigCoresOnly);
}

/**
 * @brief      Returns number of CPU logical cores on Linux/Windows (on other OSes it simply relies on the original
 * parallel API of choice, which uses the 'all' logical cores). call function with 'false' to get #logical cores of
 * all types call function with 'true' to get #logical 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  bigCoresOnly Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of logical CPU cores.
 */
inline int getNumberOfLogicalCPUCores(bool bigCoresOnly = false) {
    return ov::get_number_of_logical_cpu_cores(bigCoresOnly);
}

/**
 * @brief      Checks whether CPU supports SSE 4.2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is SSE 4.2 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_sse42;

/**
 * @brief      Checks whether CPU supports AVX capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx;

/**
 * @brief      Checks whether CPU supports AVX2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX2 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx2;

/**
 * @brief      Checks whether CPU supports AVX2_VNNI capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX2_VNNI instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx2_vnni;

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F (foundation) instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512f;

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core;

/**
 * @brief      Checks whether CPU supports AVX 512 VNNI capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ, AVX512_VNNI instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core_vnni;

/**
 * @brief      Checks whether CPU supports BFloat16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAVX512_BF16 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_bfloat16;

/**
 * @brief      Checks whether CPU supports fp16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAVX512_FP16 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core_fp16;

/**
 * @brief      Checks whether CPU supports AMX int8 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_INT8 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core_amx_int8;

/**
 * @brief      Checks whether CPU supports AMX bf16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_BF16 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core_amx_bf16;

/**
 * @brief      Checks whether CPU supports AMX capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_INT8 or tAMX_BF16 instructions are available, `false` otherwise
 */
using ov::with_cpu_x86_avx512_core_amx;

/**
 * @brief      Checks whether CPU mapping Available
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is CPU mapping is available, `false` otherwise
 */
using ov::is_cpu_map_available;

/**
 * @brief      Get number of numa nodes
 * @ingroup    ie_dev_api_system_conf
 * @return     Number of numa nodes
 */
using ov::get_num_numa_nodes;

/**
 * @brief      Get number of sockets
 * @ingroup    ie_dev_api_system_conf
 * @return     Number of sockets
 */
using ov::get_num_sockets;

/**
 * @brief      Returns number of CPU cores on Linux/Windows
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  plugin_task plugin task.
 * @return     Number of CPU cores with core_type.
 */
using ov::get_proc_type_table;

/**
 * @brief      Returns original number of CPU cores on Linux/Windows
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  plugin_task plugin task.
 * @return     Number of original CPU cores with core_type.
 */
using ov::get_org_proc_type_table;

/**
 * @brief      Get and reserve available cpu ids
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  streams_info_table streams information table.
 * @param[in]  stream_processors processors grouped in stream
 * @param[in]  cpu_status set cpu status
 */
using ov::reserve_available_cpus;

/**
 * @brief      Set flag bit 'Used' of CPU
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  cpu_ids cpus in cup_mapping.
 * @param[in]  used flag bit
 */
using ov::set_cpu_used;

/**
 * @brief      Get socket id by current numa node id
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  numa_node_id numa node id
 * @return     socket id
 */
using ov::get_socket_by_numa_node;

/**
 * @brief      This enum contains definition of each columns in processor type table which bases on cpu core types. Will
 * extend to support other CPU core type like ARM.
 *
 * The following are two example of processor type table.
 *  1. Processor table of two socket CPUs XEON server
 *
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC
 *     96            48                 0                       48          // Total number of two sockets
 *     48            24                 0                       24          // Number of socket one
 *     48            24                 0                       24          // Number of socket two
 *
 * 2. Processor table of one socket CPU desktop
 *
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC
 *     32            8                 16                       8           // Total number of one socket
 */
using ov::ColumnOfProcessorTypeTable;

/**
 * @brief      This enum contains definition of each columns in CPU mapping table which use processor id as index.
 *
 * GROUP_ID is generated according to the following rules.
 *  1. If one MAIN_CORE_PROC and one HYPER_THREADING_PROC are based on same Performance-cores, they are in one group.
 *  2. If some EFFICIENT_CORE_PROC share one L2 cachle, they are in one group.
 *  3. There are no duplicate group IDs in the system
 *
 * The following is the example of CPU mapping table.
 *  1. Four processors of two Pcore
 *  2. Four processors of four Ecores shared L2 cache
 *
 *  PROCESSOR_ID | SOCKET_ID | CORE_ID | CORE_TYPE | GROUP_ID | Used
 *       0             0          0          3          0        0
 *       1             0          0          1          0        0
 *       2             0          1          3          1        0
 *       3             0          1          1          1        0
 *       4             0          2          2          2        0
 *       5             0          3          2          2        0
 *       6             0          4          2          2        0
 *       7             0          5          2          2        0
 */
using ov::ColumnOfCPUMappingTable;

/**
 * @brief      definition of CPU_MAP_USED_FLAG column in CPU mapping table.
 */
using ov::ProcessorUseStatus;

}  // namespace InferenceEngine
