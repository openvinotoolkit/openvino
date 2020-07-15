// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hddl_metrics.hpp
 */

#pragma once

#include "common_utils.hpp"

#include "ie_plugin_config.hpp"
#include "ie_api.h"

namespace InferenceEngine {

namespace Metrics {

/**
* @brief Metric to get a int of the device number, String value is METRIC_MYRIAD_DEVICE_NUM
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_NUM, int);

/**
* @brief Metric to get a std::vector<std::string> of device names, String value is METRIC_MYRIAD_DEVICE_NAME
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_NAME, std::vector<std::string>);

/**
* @brief  Metric to get a std::vector<float> of device thermal, String value is METRIC_MYRIAD_DEVICE_THERMAL
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_THERMAL, std::vector<float>);

/**
* @brief  Metric to get a std::vector<uint32> of device ids, String value is METRIC_MYRIAD_DEVICE_ID
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_ID, std::vector<unsigned int>);

/**
* @brief  Metric to get a std::vector<int> of device subclasses, String value is METRIC_MYRIAD_DEVICE_SUBCLASS
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_SUBCLASS, std::vector<int>);

/**
* @brief  Metric to get a std::vector<uint32> of device total memory, String value is METRIC_MYRIAD_MEMORY_TOTAL
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_MEMORY_TOTAL, std::vector<unsigned int>);

/**
* @brief  Metric to get a std::vector<uint32> of device used memory, String value is METRIC_MYRIAD_DEVICE_MEMORY_USED
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_MEMORY_USED, std::vector<unsigned int>);

/**
* @brief  Metric to get a std::vector<float> of device utilization, String value is METRIC_MYRIAD_DEVICE_UTILIZATION
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_UTILIZATION, std::vector<float>);

/**
* @brief  Metric to get a std::vector<std::string> of stream ids, String value is METRIC_MYRIAD_DEVICE_STREAM_ID
*/
DECLARE_MYRIAD_METRIC_KEY(STREAM_ID, std::vector<std::string>);

/**
* @brief  Metric to get a std::vector<std::string> of device tags, String value is METRIC_MYRIAD_DEVICE_TAG
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_TAG, std::vector<std::string>);

/**
* @brief  Metric to get a std::vector<int> of group ids, String value is METRIC_MYRIAD_GROUP_ID
*/
DECLARE_MYRIAD_METRIC_KEY(GROUP_ID, std::vector<int>);

/**
* @brief  Metric to get a int number of device be using for group, String value is METRIC_MYRIAD_DEVICE_GROUP_USING_NUM
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_GROUP_USING_NUM, int);

/**
* @brief  Metric to get a int number of total device, String value is METRIC_MYRIAD_DEVICE_TOTAL_NUM
*/
DECLARE_MYRIAD_METRIC_KEY(DEVICE_TOTAL_NUM, int);

}  // namespace Metrics

}  // namespace InferenceEngine
