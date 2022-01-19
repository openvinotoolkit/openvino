// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ProfilingInfo objects that contains performance
 *        metric for single node
 *
 * @file openvino/runtime/profiling_info.hpp
 */
#pragma once

#include <chrono>
#include <string>

namespace ov {
namespace runtime {
/**
 * @struct ProfilingInfo
 * @brief Represents basic inference profiling information per operation.
 *
 * If the operation is executed using tiling, the sum time per each tile is indicated as the total execution time.
 * Due to parallel execution, the total execution time for all nodes might be greater than the total inference time.
 */
struct ProfilingInfo {
    /**
     * @brief Defines the general status of the node
     */
    enum class Status {
        NOT_RUN,        //!< A node is not executed
        OPTIMIZED_OUT,  //!< A node is optimized out during graph optimization phase
        EXECUTED        //!< A node is executed
    };

    /**
     * @brief Defines a node status
     */
    Status status;

    /**
     * @brief The absolute time in microseconds that the node ran (in total)
     */
    std::chrono::microseconds real_time;
    /**
     * @brief The net host cpu time that the node ran
     */
    std::chrono::microseconds cpu_time;

    /**
     * @brief A name of node
     */
    std::string node_name;

    /**
     * @brief An execution type of unit
     */
    std::string exec_type;

    /**
     * @brief A node type
     */
    std::string node_type;
};
}  // namespace runtime
}  // namespace ov