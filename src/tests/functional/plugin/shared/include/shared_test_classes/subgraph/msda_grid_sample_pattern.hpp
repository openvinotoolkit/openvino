// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

#include "shared_test_classes/subgraph/msda_pattern.hpp"

namespace ov {
namespace test {

// VariadicSplit and Gather based formulation of multi-scale deformable
// attention, as exported for Deformable-DETR, GroundingDINO and RT-DETR: the
// levels are VariadicSplit outputs of value and take their locations with a
// scalar Gather index.
class MSDAGridSamplePattern : public SubgraphBaseTest, public testing::WithParamInterface<MSDAPatternParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
