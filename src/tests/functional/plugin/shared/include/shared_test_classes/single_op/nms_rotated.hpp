// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    std::vector<ov::test::InputShape>,  // Input shapes
    ov::test::ElementType,              // Boxes and scores input precisions
    ov::test::ElementType,              // Max output boxes input precisions
    ov::test::ElementType,              // Thresholds precisions
    ov::test::ElementType,              // Output type
    int64_t,                            // Max output boxes per class
    float,                              // IOU threshold
    float,                              // Score threshold
    bool,                               // Sort result descending
    bool,                               // Clockwise
    bool,                               // Is 1st input constant
    bool,                               // Is 2nd input constant
    bool,                               // Is 3rd input constant
    bool,                               // Is 4th input constant
    bool,                               // Is 5th input constant
    ov::AnyMap,                         // Additional configuration
    std::string                         // Device name
> NmsRotatedParams;

class NmsRotatedOpTest : public testing::WithParamInterface<NmsRotatedParams>,
                         public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsRotatedParams>& obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    int64_t m_max_out_boxes_per_class;
    float m_iou_threshold;
    float m_score_threshold;
};

} // namespace LayerTestsDefinitions
