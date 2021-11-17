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
        std::vector<std::vector<size_t>>, // indices
        bool                 // with_weights
> embeddingBagPackedSumParams;

typedef std::tuple<
        embeddingBagPackedSumParams,
        ov::test::ElementType, // embedding table
        ov::test::ElementType, // indices
        LayerTestsUtils::TargetDevice> embeddingBagPackedSumLayerTestParamsSet;


class EmbeddingBagPackedSumLayerTest : public testing::WithParamInterface<embeddingBagPackedSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
