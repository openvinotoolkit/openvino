// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines set of macro to safely set plugin and executable network metric values
 * @file ie_metric_helpers.hpp
 */

#pragma once

#include <type_traits>
#include <utility>

/**
 * @cond
 */

namespace InferenceEngine {
namespace Metrics {

template <typename T>
struct MetricType;

#define DECLARE_METRIC_KEY_IMPL(name, ...) \
    struct name {};                        \
    template <>                            \
    struct MetricType<name> {              \
        using type = __VA_ARGS__;          \
    }

}  // namespace Metrics
}  // namespace InferenceEngine

/**
 * @endcond
 */

/**
 * @def        IE_SET_METRIC_RETURN(name, ...)
 * @ingroup    ie_dev_api
 * @brief      Return metric value with specified @p name and arguments `...`. Example:
 * @code
 * IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
 * @endcode
 *
 * @param      name  The metric name
 * @param      ...   A metric value
 *
 * @return     A metric value wrapped with Parameter and returned to a calling function
 */
#define IE_SET_METRIC_RETURN(name, ...)                                                                       \
    typename ::InferenceEngine::Metrics::MetricType<::InferenceEngine::Metrics::name>::type _##name##_value = \
        __VA_ARGS__;                                                                                          \
    return _##name##_value

#include "ie_plugin_config.hpp"
