// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ExecGraphInfoSerialization {
/**
* @brief Executable Graph Info is represented in ICNNNetwork format with general CNNLayer nodes inside
*        including connections between the nodes. Each node describes an executable hardware-specific
*        primitive and stores its parameters within CNNLayer::params map.
*        There is a list of general keys for the parameters map.
*/

/**
 * @brief A general key for CNNLayer::params map. Used to get a string of layer names separated by a comma
 *        from the original IR, which were fused/merged to the current executable primitive.
 */
static const char ORIGIN_NAMES[] = "originalFusedLayersNames";
/**
 * @brief A general key for CNNLayer::params map. Used to get a type of the executable primitive.
 */
static const char IMPL_TYPE[] = "primitiveType";
/**
 * @brief A general key for CNNLayer::params map. Used to get a precision of the executable primitive.
 */
static const char PRECISION[] = "precision";
/**
 * @brief A general key for CNNLayer::params map. Used to get value of execution time of the executable primitive.
 */
static const char PERF_COUNTER[] = "execTimeMcs";
}  // namespace ExecGraphInfoSerialization