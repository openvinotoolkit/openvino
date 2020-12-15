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
        std::vector<size_t>, // indices
        std::vector<size_t>, // offsets
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
> embeddingBagOffsetsSumParams;

typedef std::tuple<
        embeddingBagOffsetsSumParams,
        InferenceEngine::Precision, // embedding table
        InferenceEngine::Precision, // indices
        LayerTestsUtils::TargetDevice> embeddingBagOffsetsSumLayerTestParamsSet;

class EmbeddingBagOffsetsSumLayerTest : public testing::WithParamInterface<embeddingBagOffsetsSumLayerTestParamsSet>,
            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
