// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include <ie_metric_helpers.hpp>
#include <ie_core.hpp>

#define IE_SET_METRIC(key, name,  ...)                                                            \
    typename ::InferenceEngine::Metrics::MetricType<::InferenceEngine::Metrics::key>::type name = \
        __VA_ARGS__;

#define RETURN_MOCK_VALUE(value) \
    InvokeWithoutArgs([value](){return value;})

//  getMetric will return a fake ov::Any, gmock will call ostreamer << ov::Any
//  it will cause core dump, so add this special implemented
namespace testing {
namespace internal {
    template<>
    void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os);
}
}

#define ENABLE_LOG_IN_MOCK() \
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) { \
            std::cout << stream.str() << std::endl; \
            });
