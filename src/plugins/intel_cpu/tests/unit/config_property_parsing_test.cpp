// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>

#include <gtest/gtest.h>

#include "config.h"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_cpu {
namespace {

TEST(CPUConfigReadPropertiesTests, AcceptsNumericNumStreamsAnyValues) {
    Config cfg;

    ASSERT_NO_THROW(cfg.readProperties({{ov::num_streams.name(), int64_t{1}}}));
    EXPECT_EQ(cfg.streams, 1);

    ASSERT_NO_THROW(cfg.readProperties({{ov::num_streams.name(), 2}}));
    EXPECT_EQ(cfg.streams, 2);
}

TEST(CPUConfigReadPropertiesTests, RepeatedNumericStreamPropertiesAreStable) {
    Config cfg;
    constexpr int kIterations = 4096;

    for (int i = 0; i < kIterations; ++i) {
        const int streams = (i % 8) + 1;
        const int threads = (i % 16) + 1;
        const ov::Any numStreamsValue = (i % 2 == 0) ? ov::Any{int64_t{streams}} : ov::Any{streams};
        const ov::Any numThreadsValue = (i % 2 == 0) ? ov::Any{int64_t{threads}} : ov::Any{threads};

        ASSERT_NO_THROW(cfg.readProperties({{ov::num_streams.name(), numStreamsValue},
                                            {ov::inference_num_threads.name(), numThreadsValue}}))
            << "Iteration: " << i;
        EXPECT_EQ(cfg.streams, streams);
        EXPECT_EQ(cfg.threads, threads);
    }
}

}  // namespace
}  // namespace ov::intel_cpu
