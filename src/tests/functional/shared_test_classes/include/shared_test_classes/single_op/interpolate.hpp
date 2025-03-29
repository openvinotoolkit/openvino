// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        ov::op::util::InterpolateBase::InterpolateMode,          // InterpolateMode
        ov::op::util::InterpolateBase::ShapeCalcMode,            // ShapeCalculationMode
        ov::op::util::InterpolateBase::CoordinateTransformMode,  // CoordinateTransformMode
        ov::op::util::InterpolateBase::NearestMode,              // NearestMode
        bool,                                                    // AntiAlias
        std::vector<size_t>,                                     // PadBegin
        std::vector<size_t>,                                     // PadEnd
        double,                                                  // Cube coef
        std::vector<int64_t>,                                    // Axes
        std::vector<float>                                       // Scales
> InterpolateSpecificParams;

typedef std::tuple<
        InterpolateSpecificParams,
        ov::element::Type,                 // Model type
        std::vector<InputShape>,           // Input shapes
        ov::Shape,                         // Target shapes
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
> InterpolateLayerTestParams;

class InterpolateLayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};

class Interpolate11LayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
