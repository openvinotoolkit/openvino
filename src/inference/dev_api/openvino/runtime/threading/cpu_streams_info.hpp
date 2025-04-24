// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_info.hpp
 * @brief A header file for OpenVINO CPU streams info table implementation.
 */

#pragma once

namespace ov {

/**
 * @enum       ColumnOfCpuStreamsInfoTable
 * @brief      This enum contains definition of each columns in cpu streams information table.
 *
 * The following are two example of processor type table.
 *  1. 8 streams on hybrid platform which has 4 threads per stream (TPS).
 *     1.1 2 streams (4 TPS) on physical core of Intel Performance-cores
 *     1.2 4 streams (4 TPS) on Intel Efficient-cores
 *     1.3 2 streams (4 TPS) on logic core of Intel Performance-cores
 *
 *  NUMBER_OF_STREAMS | PROC_TYPE | THREADS_PER_STREAM | STREAM_NUMA_NODE_ID | STREAM_SOCKET_ID
 *          2               1                4                    0                    0
 *          4               2                4                    0                    0
 *          2               3                4                    0                    0
 *
 * 2. 1 stream (10 TPS) on hybrid platform which has 2 threads on physical core and 8 threads on Ecore.
 *    2.1 1 streams (10 TPS) on multiple types of processors
 *    2.2 2 threads on physical core of Intel Performance-cores
 *    2.3 8 threads on Intel Efficient-cores
 *
 *  NUMBER_OF_STREAMS | PROC_TYPE | THREADS_PER_STREAM | STREAM_NUMA_NODE_ID | STREAM_SOCKET_ID
 *          1               0               10                    0                    0
 *          0               1                2                    0                    0
 *          0               2                8                    0                    0
 */
enum ColumnOfCpuStreamsInfoTable {
    NUMBER_OF_STREAMS = 0,      //!< Number of streams on specific CPU core tpye
    PROC_TYPE = 1,              //!< Core type of current streams
    THREADS_PER_STREAM = 2,     //!< Number of threads per stream of current streams
    STREAM_NUMA_NODE_ID = 3,    //!< Numa node id of processors in this row
    STREAM_SOCKET_ID = 4,       //!< Socket id of processors in this row
    CPU_STREAMS_TABLE_SIZE = 5  //!< Size of streams info table
};

}  // namespace ov
