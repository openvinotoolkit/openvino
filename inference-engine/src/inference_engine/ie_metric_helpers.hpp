// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

namespace InferenceEngine {
namespace Metrics {

template <typename T>
struct MetricType;

#define DECLARE_METRIC_KEY_IMPL(name, ...)          \
    struct name { };                                \
    template <>                                     \
    struct MetricType<name> {                       \
        using type = __VA_ARGS__;                   \
    };

}  // namespace Metrics
}  // namespace InferenceEngine

#define IE_SET_METRIC_RETURN(name, ...)                                                                                          \
    typename ::InferenceEngine::Metrics::MetricType<::InferenceEngine::Metrics::name>::type _ ## name ## _value = __VA_ARGS__;   \
    return _ ## name ## _value

#define IE_SET_METRIC(name, ...)                      \
    [&] {                                             \
        IE_SET_METRIC_RETURN(name, __VA_ARGS__);      \
    } ()

#include "ie_plugin_config.hpp"
