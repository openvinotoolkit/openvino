// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ov::test::InputShape, // input_shapes
        std::vector<size_t>, // indices
        std::vector<size_t>, // segment_ids
        size_t,              // num_segments
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
> embeddingSegmentsSumParams;

typedef std::tuple<
        embeddingSegmentsSumParams,
        ov::test::ElementType, // embedding table
        ov::test::ElementType, // indices
        LayerTestsUtils::TargetDevice> embeddingSegmentsSumLayerTestParamsSet;

class EmbeddingSegmentsSumLayerTest : public testing::WithParamInterface<embeddingSegmentsSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
