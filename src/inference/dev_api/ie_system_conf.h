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

#include "ie_api.h"

namespace InferenceEngine {

/**
 * @brief      Checks whether OpenMP environment variables are defined
 * @ingroup    ie_dev_api_system_conf
 *
 * @param[in]  includeOMPNumThreads  Indicates if the omp number threads is included
 * @return     `True` if any OpenMP environment variable is defined, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) checkOpenMpEnvVars(bool includeOMPNumThreads = true);

/**
 * @brief      Returns available CPU NUMA nodes (on Linux, and Windows [only with TBB], single node is assumed on all
 * other OSes)
 * @ingroup    ie_dev_api_system_conf
 * @return     NUMA nodes
 */
INFERENCE_ENGINE_API_CPP(std::vector<int>) getAvailableNUMANodes();

/**
 * @brief      Returns available CPU cores types (on Linux, and Windows) and ONLY with TBB, single core type is assumed
 * otherwise
 * @ingroup    ie_dev_api_system_conf
 * @return     Vector of core types
 */
INFERENCE_ENGINE_API_CPP(std::vector<int>) getAvailableCoresTypes();

/**
 * @brief      Returns number of CPU physical cores on Linux/Windows (which is considered to be more performance
 * friendly for servers) (on other OSes it simply relies on the original parallel API of choice, which usually uses the
 * logical cores). call function with 'false' to get #phys cores of all types call function with 'true' to get #phys
 * 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  bigCoresOnly Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of physical CPU cores.
 */
INFERENCE_ENGINE_API_CPP(int) getNumberOfCPUCores(bool bigCoresOnly = false);

/**
 * @brief      Returns number of CPU logical cores on Linux/Windows (on other OSes it simply relies on the original
 * parallel API of choice, which uses the 'all' logical cores). call function with 'false' to get #logical cores of
 * all types call function with 'true' to get #logical 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  bigCoresOnly Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of logical CPU cores.
 */
INFERENCE_ENGINE_API_CPP(int) getNumberOfLogicalCPUCores(bool bigCoresOnly = false);

/**
 * @brief      Checks whether CPU supports SSE 4.2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is SSE 4.2 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_sse42();

/**
 * @brief      Checks whether CPU supports AVX capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx();

/**
 * @brief      Checks whether CPU supports AVX2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX2 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx2();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F (foundation) instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512f();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core();

/**
 * @brief      Checks whether CPU supports AVX 512 VNNI capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ, AVX512_VNNI instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core_vnni();

/**
 * @brief      Checks whether CPU supports BFloat16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAVX512_BF16 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_bfloat16();

/**
 * @brief      Checks whether CPU supports AMX int8 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_INT8 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core_amx_int8();

/**
 * @brief      Checks whether CPU supports AMX bf16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_BF16 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core_amx_bf16();

/**
 * @brief      Checks whether CPU supports AMX capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAMX_INT8 or tAMX_BF16 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core_amx();

/**
 * @enum       column_of_processor_type_table
 * @brief      This enum contains defination of each columns in processor type table which bases on cpu core types. Will
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
typedef enum {
    ALL_PROC = 0,              //!< All processors, regardless of backend cpu
    MAIN_CORE_PROC = 1,        //!< Processor based on physical core of Intel Performance-cores
    EFFICIENT_CORE_PROC = 2,   //!< Processor based on Intel Efficient-cores
    HYPER_THREADING_PROC = 3,  //!< Processor based on logical core of Intel Performance-cores
    PROC_TYPE_TABLE_SIZE = 4   //!< Size of processor type table
} column_of_processor_type_table;

/**
 * @enum       column_of_cpu_mapping_table
 * @brief      This enum contains defination of each columns in CPU mapping table which use processor id as index.
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
typedef enum {
    CPU_MAP_PROCESSOR_ID = 0,  //!< column for processor id of the processor
    CPU_MAP_SOCKET_ID = 1,     //!< column for socket id of the processor
    CPU_MAP_CORE_ID = 2,       //!< column for hardware core id of the processor
    CPU_MAP_CORE_TYPE = 3,     //!< column for CPU core type corresponding to the processor
    CPU_MAP_GROUP_ID = 4,      //!< column for group id to the processor. Processors in one group have dependency.
    CPU_MAP_USED_FLAG = 5,     //!< column for resource management of the processor
    CPU_MAP_TABLE_SIZE = 6     //!< Size of CPU mapping table
} column_of_cpu_mapping_table;

}  // namespace InferenceEngine
