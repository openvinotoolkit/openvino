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
        std::vector<size_t>, // offsets
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
    > embeddingBagOffsetsSumParams;

typedef std::tuple<
        embeddingBagOffsetsSumParams,
        InferenceEngine::Precision,
        LayerTestsUtils::TargetDevice> embeddingBagOffsetsSumLayerTestParamsSet;

namespace LayerTestsDefinitions {

class EmbeddingBagOffsetsSumLayerTest : public testing::WithParamInterface<embeddingBagOffsetsSumLayerTestParamsSet>,
            public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
