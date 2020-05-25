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
        std::vector<std::vector<size_t>>, // indices
        bool                 // with_weights
    > embeddingBagPackedSumParams;

typedef std::tuple<
        embeddingBagPackedSumParams,
        InferenceEngine::Precision,
        LayerTestsUtils::TargetDevice> embeddingBagPackedSumLayerTestParamsSet;

namespace LayerTestsDefinitions {

class EmbeddingBagPackedSumLayerTest : public testing::WithParamInterface<embeddingBagPackedSumLayerTestParamsSet>,
            public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
