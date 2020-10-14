// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "functional_test_utils/layer_test_utils.hpp"

typedef std::tuple<
        std::vector<size_t>, // emb_table_shape
        std::vector<size_t>, // indices
        std::vector<size_t>, // segment_ids
        size_t,              // num_segments
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
    > embeddingSegmentsSumParams;

typedef std::tuple<
        embeddingSegmentsSumParams,
        InferenceEngine::Precision, // embedding table
        InferenceEngine::Precision, // indices
        LayerTestsUtils::TargetDevice> embeddingSegmentsSumLayerTestParamsSet;

namespace LayerTestsDefinitions {

class EmbeddingSegmentsSumLayerTest : public testing::WithParamInterface<embeddingSegmentsSumLayerTestParamsSet>,
            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
