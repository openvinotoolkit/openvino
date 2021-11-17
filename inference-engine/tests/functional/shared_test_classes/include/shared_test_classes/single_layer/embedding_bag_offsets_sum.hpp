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
        std::vector<size_t>, // offsets
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
> embeddingBagOffsetsSumParams;

typedef std::tuple<
        embeddingBagOffsetsSumParams,
        ov::test::ElementType, // embedding table
        ov::test::ElementType, // indices
        LayerTestsUtils::TargetDevice> embeddingBagOffsetsSumLayerTestParamsSet;

class EmbeddingBagOffsetsSumLayerTest : public testing::WithParamInterface<embeddingBagOffsetsSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
