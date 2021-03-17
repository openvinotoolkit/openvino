// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<size_t>, // emb_table_shape
        std::vector<std::vector<size_t>>, // indices
        bool                 // with_weights
> embeddingBagPackedSumParams;

typedef std::tuple<
        embeddingBagPackedSumParams,
        InferenceEngine::Precision, // embedding table
        InferenceEngine::Precision, // indices
        LayerTestsUtils::TargetDevice> embeddingBagPackedSumLayerTestParamsSet;


class EmbeddingBagPackedSumLayerTest : public testing::WithParamInterface<embeddingBagPackedSumLayerTestParamsSet>,
            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
