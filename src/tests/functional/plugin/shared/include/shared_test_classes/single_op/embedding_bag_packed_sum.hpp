// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<std::vector<size_t>>,  // indices
        bool                               // with_weights
> embeddingBagPackedSumParams;

typedef std::tuple<
        embeddingBagPackedSumParams,
        std::vector<InputShape>, // input shapes
        ov::element::Type,       // embedding table
        ov::element::Type,       // indices
        std::string> embeddingBagPackedSumLayerTestParamsSet;

class EmbeddingBagPackedSumLayerTest : public testing::WithParamInterface<embeddingBagPackedSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
