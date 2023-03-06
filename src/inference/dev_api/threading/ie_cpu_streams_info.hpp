// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_cpu_streams_info.hpp
 * @brief A header file for Inference Engine CPU streams info table implementation.
 */

#pragma once

namespace InferenceEngine {

/**
 * @enum       column_of_cpu_streams_info_table
 * @brief      This enum contains definition of each columns in cpu streams information table.
 *
 * The following are two example of processor type table.
 *  1. 8 streams on hybrid platform which has 4 threads per stream (TPS).
 *
 *  NUMBER_OF_STREAMS | PROC_TYPE | THREADS_PER_STREAM
 *          2               1                4          // 2 streams (4 TPS) on physical core of Intel Performance-cores
 *          4               2                4          // 4 streams (4 TPS) on Intel Efficient-cores
 *          2               3                4          // 2 streams (4 TPS) on logic core of Intel Performance-cores
 *
 * 2. 1 stream (10 TPS) on hybrid platform which has 2 threads on physical core and 8 threads on Ecore.
 *
 *  NUMBER_OF_STREAMS | PROC_TYPE | THREADS_PER_STREAM
 *          1               0               10          // 1 streams (10 TPS) on multiple types of processors
 *          0               1                2          // 2 threads on physical core of Intel Performance-cores
 *          0               2                8          // 8 threads on Intel Efficient-cores
 */
typedef enum {
    NUMBER_OF_STREAMS = 0,      //!< Number of streams on specific CPU core tpye
    PROC_TYPE = 1,              //!< Core type of current streams
    THREADS_PER_STREAM = 2,     //!< Number of threads per stream of current streams
    CPU_STREAMS_TABLE_SIZE = 3  //!< Size of streams info table
} column_of_cpu_streams_info_table;

}  // namespace InferenceEngine
