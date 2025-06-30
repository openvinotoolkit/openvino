// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        ov::test::utils::PoolingTypes,  // Pooling type, max or avg
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Stride
        std::vector<size_t>,            // Pad begin
        std::vector<size_t>,            // Pad end
        ov::op::RoundingType,           // Rounding type
        ov::op::PadType,                // Pad type
        bool                            // Exclude pad
> poolSpecificParams;

typedef std::tuple<
        poolSpecificParams,
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shape
        std::string                     // Device name
> poolLayerTestParamsSet;

typedef std::tuple<
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Stride
        std::vector<size_t>,            // Dilation
        std::vector<size_t>,            // Pad begin
        std::vector<size_t>,            // Pad end
        ov::element::Type,              // Index element type
        int64_t,                        // Axis
        ov::op::RoundingType,           // Rounding type
        ov::op::PadType                 // Pad type
> maxPoolV8SpecificParams;

typedef std::tuple<
        maxPoolV8SpecificParams,
        ov::element::Type,              // Model type
        std::vector<InputShape>,        // Input shape
        std::string                     // Device name
> maxPoolV8LayerTestParamsSet;

class PoolingLayerTest : public testing::WithParamInterface<poolLayerTestParamsSet>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

class MaxPoolingV8LayerTest : public testing::WithParamInterface<maxPoolV8LayerTestParamsSet>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<maxPoolV8LayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
