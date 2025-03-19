// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file openvino/runtime/system_conf.hpp
 */

#pragma once

#include <vector>

#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief      Checks whether OpenMP environment variables are defined
 * @ingroup    ov_dev_api_system_conf
 *
 * @param[in]  include_omp_num_threads  Indicates if the omp number threads is included
 * @return     `True` if any OpenMP environment variable is defined, `false` otherwise
 */
OPENVINO_RUNTIME_API bool check_open_mp_env_vars(bool include_omp_num_threads = true);

/**
 * @brief      Returns available CPU NUMA nodes (on Linux, and Windows [only with TBB], single node is assumed on all
 * other OSes)
 * @ingroup    ov_dev_api_system_conf
 * @return     NUMA nodes
 */
OPENVINO_RUNTIME_API std::vector<int> get_available_numa_nodes();

/**
 * @brief      Returns available CPU cores types (on Linux, and Windows) and ONLY with TBB, single core type is assumed
 * otherwise
 * @ingroup    ov_dev_api_system_conf
 * @return     Vector of core types
 */
OPENVINO_RUNTIME_API std::vector<int> get_available_cores_types();

/**
 * @brief      Returns number of CPU physical cores on Linux/Windows (which is considered to be more performance
 * friendly for servers) (on other OSes it simply relies on the original parallel API of choice, which usually uses the
 * logical cores). call function with 'false' to get #phys cores of all types call function with 'true' to get #phys
 * 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  big_cores_only Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of physical CPU cores.
 */
OPENVINO_RUNTIME_API int get_number_of_cpu_cores(bool big_cores_only = false);

/**
 * @brief      Returns number of CPU logical cores on Linux/Windows (on other OSes it simply relies on the original
 * parallel API of choice, which uses the 'all' logical cores). call function with 'false' to get #logical cores of
 * all types call function with 'true' to get #logical 'Big' cores number of 'Little' = 'all' - 'Big'
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  big_cores_only Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of logical CPU cores.
 */
OPENVINO_RUNTIME_API int get_number_of_logical_cpu_cores(bool big_cores_only = false);

/**
 * @brief      Returns number of blocked CPU cores. Please note that this is a temporary interface for performance
 * optimization on a specific platform. May be removed in future release.
 * @ingroup    ov_dev_api_system_conf
 * @return     Number of blocked CPU cores.
 */
OPENVINO_RUNTIME_API int get_number_of_blocked_cores();

/**
 * @brief      Checks whether CPU supports SSE 4.2 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is SSE 4.2 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_sse42();

/**
 * @brief      Checks whether CPU supports ARM NEON FP16 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is ARM NEON FP16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_neon_fp16();

/**
 * @brief      Checks whether CPU supports ARM SVE capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` if ARM SVE instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_sve();

/**
 * @brief      Checks whether CPU supports AVX capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx();

/**
 * @brief      Checks whether CPU supports AVX2 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX2 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx2();

/**
 * @brief      Checks whether CPU supports AVX2_VNNI capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX2_VNNI instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx2_vnni();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX512F (foundation) instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512f();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core();

/**
 * @brief      Checks whether CPU supports AVX 512 VNNI capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ, AVX512_VNNI instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_vnni();

/**
 * @brief      Checks whether CPU supports BFloat16 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAVX512_BF16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_bfloat16();

/**
 * @brief      Checks whether CPU supports fp16 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAVX512_FP16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_fp16();

/**
 * @brief      Checks whether CPU supports AMX int8 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAMX_INT8 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_amx_int8();

/**
 * @brief      Checks whether CPU supports AMX bf16 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAMX_BF16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_amx_bf16();

/**
 * @brief      Checks whether CPU supports AMX fp16 capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAMX_FP16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_amx_fp16();

/**
 * @brief      Checks whether CPU supports AMX capability
 * @ingroup    ov_dev_api_system_conf
 * @return     `True` is tAMX_INT8 or tAMX_BF16 instructions are available, `false` otherwise
 */
OPENVINO_RUNTIME_API bool with_cpu_x86_avx512_core_amx();

/**
 * @brief      Get number of numa nodes
 * @ingroup    ov_dev_api_system_conf
 * @return     Number of numa nodes
 */
OPENVINO_RUNTIME_API int get_num_numa_nodes();

/**
 * @brief      Get number of sockets
 * @ingroup    ov_dev_api_system_conf
 * @return     Number of sockets
 */
OPENVINO_RUNTIME_API int get_num_sockets();

/**
 * @brief      Get numa node id of cpu_id
 * @ingroup    ov_dev_api_system_conf
 * @return     Numa node id
 */
OPENVINO_RUNTIME_API int get_numa_node_id(int cpu_id);

/**
 * @brief      Returns a table of number of processor types on Linux/Windows
 * @ingroup    ov_dev_api_system_conf
 * @return     A table about number of CPU cores of different types defined with ColumnOfProcessorTypeTable
 * The following are two example of processor type table.
 *  1. Processor table of two socket CPUs XEON server
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC
 *     96            48                 0                       48          // Total number of two sockets
 *     48            24                 0                       24          // Number of socket one
 *     48            24                 0                       24          // Number of socket two
 *
 * 2. Processor table of one socket CPU desktop
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC
 *     32            8                 16                       8           // Total number of one socket
 */
OPENVINO_RUNTIME_API std::vector<std::vector<int>> get_proc_type_table();

/**
 * @brief      Returns the socket ID in cpu mapping table of the currently running thread.
 * @ingroup    ov_dev_api_system_conf
 * @return     socket ID in cpu mapping
 */
OPENVINO_RUNTIME_API int get_current_socket_id();

/**
 * @brief      Returns the numa node ID in cpu mapping table of the currently running thread.
 * @ingroup    ov_dev_api_system_conf
 * @return     numa node ID in cpu mapping
 */
OPENVINO_RUNTIME_API int get_current_numa_node_id();

/**
 * @brief      Returns a table of original number of processor types without filtering other plugins occupying CPU
 * resources. The difference from get_proc_type_table: This is used to get the configuration of current machine. For
 * example, GPU plugin occupies all Pcores, there is only one type core in proc_type_table from get_proc_type_table().
 * If user wants to get the real configuration of this machine which should be got from get_org_proc_type_table.
 * @ingroup    ov_dev_api_system_conf
 * @return     A table about number of CPU cores of different types defined with ColumnOfProcessorTypeTable
 */
OPENVINO_RUNTIME_API std::vector<std::vector<int>> get_org_proc_type_table();

/**
 * @enum       ColumnOfProcessorTypeTable
 * @brief      This enum contains definition of each columns in processor type table which bases on cpu core types. Will
 * extend to support other CPU core type like ARM.
 *
 * The following are two example of processor type table.
 *  1. Processor table of 4 numa nodes and 2 socket server
 *
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC | PROC_NUMA_NODE_ID | PROC_SOCKET_ID
 *     96            48                 0                       48                  -1                 -1
 *     24            12                 0                       12                   0                  0
 *     24            12                 0                       12                   1                  0
 *     24            12                 0                       12                   2                  1
 *     24            12                 0                       12                   3                  1
 *
 * 2. Processor table of 1 numa node desktop
 *
 *  ALL_PROC | MAIN_CORE_PROC | EFFICIENT_CORE_PROC | HYPER_THREADING_PROC | PROC_NUMA_NODE_ID | PROC_SOCKET_ID
 *     32            8                 16                       8                   -1                 -1
 */
enum ColumnOfProcessorTypeTable {
    ALL_PROC = 0,              //!< All processors, regardless of backend cpu
    MAIN_CORE_PROC = 1,        //!< Processor based on physical core of Intel Performance-cores
    EFFICIENT_CORE_PROC = 2,   //!< Processor based on Intel Efficient-cores
    HYPER_THREADING_PROC = 3,  //!< Processor based on logical core of Intel Performance-cores
    PROC_NUMA_NODE_ID = 4,     //!< Numa node id of processors in this row
    PROC_SOCKET_ID = 5,        //!< Socket id of processors in this row
    PROC_TYPE_TABLE_SIZE = 6   //!< Size of processor type table
};

/**
 * @enum       ProcessorUseStatus
 * @brief      Definition of CPU_MAP_USED_FLAG column in CPU mapping table.
 */
enum ProcessorUseStatus {
    CPU_BLOCKED = -100,  //!< Processor is blocked to use
    NOT_USED = -1,       //!< Processor is not bound to thread
    CPU_USED = 1,        //!< CPU is in using
};

/**
 * @brief      Get and reserve available cpu ids
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  streams_info_table streams information table.
 * @param[in]  stream_processors processors grouped in stream which is used in core binding in cpu streams executor
 * @param[in]  cpu_status set cpu status
 */
OPENVINO_RUNTIME_API void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                                                 std::vector<std::vector<int>>& stream_processors,
                                                 const int cpu_status = NOT_USED);

/**
 * @brief      Set CPU_MAP_USED_FLAG of cpu_mapping
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  cpu_ids cpus in cpu_mapping.
 * @param[in]  used update CPU_MAP_USED_FLAG of cpu_mapping with this flag bit
 */
OPENVINO_RUNTIME_API void set_cpu_used(const std::vector<int>& cpu_ids, const int used);

/**
 * @brief      Get original socket id by current socket id, the input socket id is recalculated after filtering (like
 * numactl), while the original socket id is the original id before filtering
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  socket_id socket id
 * @return     socket id
 */
OPENVINO_RUNTIME_API int get_org_socket_id(int socket_id);

/**
 * @brief      Get original numa node id by current numa node id, the input numa node id is recalculated after filtering
 * (like numactl), while the original numa node id is the original id before filtering
 * @ingroup    ov_dev_api_system_conf
 * @param[in]  numa_node_id numa node id
 * @return     numa node id
 */
OPENVINO_RUNTIME_API int get_org_numa_id(int numa_node_id);

/**
 * @enum       ColumnOfCPUMappingTable
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
 *  PROCESSOR_ID | NUMA_NODE_ID | SOCKET_ID | CORE_ID | CORE_TYPE | GROUP_ID | Used
 *       0               0            0          0          3          0        0
 *       1               0            0          0          1          0        0
 *       2               0            0          1          3          1        0
 *       3               0            0          1          1          1        0
 *       4               0            0          2          2          2        0
 *       5               0            0          3          2          2        0
 *       6               0            0          4          2          2        0
 *       7               0            0          5          2          2        0
 */
enum ColumnOfCPUMappingTable {
    CPU_MAP_PROCESSOR_ID = 0,  //!< column for processor id of the processor
    CPU_MAP_NUMA_NODE_ID = 1,  //!< column for node id of the processor
    CPU_MAP_SOCKET_ID = 2,     //!< column for socket id of the processor
    CPU_MAP_CORE_ID = 3,       //!< column for hardware core id of the processor
    CPU_MAP_CORE_TYPE = 4,     //!< column for CPU core type corresponding to the processor
    CPU_MAP_GROUP_ID = 5,      //!< column for group id to the processor. Processors in one group have dependency.
    CPU_MAP_USED_FLAG = 6,     //!< column for resource management of the processor
    CPU_MAP_TABLE_SIZE = 7     //!< Size of CPU mapping table
};

}  // namespace ov
