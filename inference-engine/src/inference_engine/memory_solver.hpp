// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header provides a declaration of MemorySolver utility class
 * @file
 */
#pragma once

#include "ie_api.h"

#include <vector>
#include <map>

namespace InferenceEngine {

/**
 * @brief Helps to solve issue of optimal memory allocation only for particular
 *        execution order.
 *
 * It works with abstract data description where
 * - Node is index in execution order
 * - Edge is Box object with size and start-finish indexes (live time)
 *
 * Example:
 *
 * Mem
 *  |        |____|             Box {4, 5}
 *  |  |_____________|          Box {2, 6}
 *  |     |____|                Box {3, 4}
 *  |  |____|                   Box {2, 3}
 *  |              |____|       Box {6, 7}
 *  |_____________________________________
 *   1  2  3  4  5  6  7  8  9  ExecOrder
 *
 *  Boxes which has an ExecOrder-axis intersection should have no Mem-axis intersections.
 *  The goal is to define a minimal required memory blob to store all boxes with such
 *  constraints and specify all corresponfing position on Mem axis(through offset field).
 *
 *  NOTE!
 *  Exec order is predefined.
 */

class INFERENCE_ENGINE_API_CLASS(MemorySolver) {
public:
    /** @brief Representation of edge (size and live time)*/
    struct Box {
        /** Execution order index of first use. The data will be produced here. */
        int start;

        /**
         * The execution order index of last use. After that data will be released.
         * -1 is a reserved value for "till to end". The data will be alive to very
         * end of execution.
         */
        int finish;

        /** Size of data. In abstract unit of measure (byte, simd, cache line, ...) */
        int size;

        /** Box identifier, unique for each box. Will be used to querying calculated offset. */
        int id;
    };

    explicit MemorySolver(const std::vector<Box>& boxes);

    /**
     * @brief Solve memory location with maximal reuse.
     * @return Size of common memory blob required for storing all
     */
    int solve();

    /** Provides calculated offset for specified box id */
    int getOffset(int id) const;

    /** Additional info. Max sum of box sizes required for any time stamp. */
    int maxDepth();
    /** Additional info. Max num of boxes required for any time stamp. */
    int maxTopDepth();

private:
    std::vector<Box> _boxes;
    std::map<int, int> _offsets;
    int _top_depth = -1;
    int _depth = -1;
    int _time_duration = -1;

    void calcDepth();
};

}  // namespace InferenceEngine
