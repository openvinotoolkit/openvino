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
        std::vector<size_t>, // indices
        std::vector<size_t>, // offsets
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
> embeddingBagOffsetsSumParams;

typedef std::tuple<
        embeddingBagOffsetsSumParams,
        std::vector<InputShape>,  // shapes
        ov::element::Type,        // model type
        ov::element::Type,        // indices type
        std::string> embeddingBagOffsetsSumLayerTestParamsSet;

class EmbeddingBagOffsetsSumLayerTest : public testing::WithParamInterface<embeddingBagOffsetsSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
