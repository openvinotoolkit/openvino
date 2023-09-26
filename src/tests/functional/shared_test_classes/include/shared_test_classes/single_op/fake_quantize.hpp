// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

// seed selected using current cloc time
#define USE_CLOCK_TIME 1
// seed started from default value, and incremented every time using big number like 9999
#define USE_INCREMENTAL_SEED 2

/**
 * redefine this seed to reproduce issue with given seed that can be read from gtest logs
 */
#define BASE_SEED   123
#define OV_SEED 123

namespace ov {
namespace test {
typedef std::tuple<
        size_t,                     // fake quantize levels
        std::vector<size_t>,        // fake quantize inputs shape
        std::vector<float>,         // fake quantize (inputLow, inputHigh, outputLow, outputHigh) or empty for random
        ov::op::AutoBroadcastSpec   // fake quantize broadcast mode
> fqSpecificParams;
typedef std::tuple<
        fqSpecificParams,
        ov::element::Type,                                         // Model type
        std::vector<InputShape>,                                   // Input shapes
        std::string                                                // Device name
> fqLayerTestParamsSet;

class FakeQuantizeLayerTest : public testing::WithParamInterface<fqLayerTestParamsSet>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fqLayerTestParamsSet>& obj);
protected:
    void SetUp() override;

    void update_seed();

    int32_t  seed = 1;
};
}  // namespace test
}  // namespace ov
