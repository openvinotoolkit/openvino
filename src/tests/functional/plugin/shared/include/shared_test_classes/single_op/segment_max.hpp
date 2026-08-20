// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using SegmentMaxSpecificParams = std::tuple<InputShape,                // data shape
                                            std::vector<int64_t>,      // segment_ids values
                                            int64_t,                   // num_segments (-1 means no num_segments input)
                                            ov::op::FillMode>;         // fill_mode

using SegmentMaxLayerTestParams = std::tuple<SegmentMaxSpecificParams,
                                             ElementType,              // data precision
                                             std::string>;             // target device

class SegmentMaxLayerTest : public testing::WithParamInterface<SegmentMaxLayerTestParams>,
                            public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SegmentMaxLayerTestParams>& obj);
    static const std::vector<SegmentMaxSpecificParams> GenerateParams();

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
