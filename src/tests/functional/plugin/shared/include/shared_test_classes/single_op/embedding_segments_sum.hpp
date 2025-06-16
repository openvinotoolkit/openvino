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
        std::vector<size_t>, // segment_ids
        size_t,              // num_segments
        size_t,              // default_index
        bool,                // with_weights
        bool                 // with_def_index
> embeddingSegmentsSumParams;

typedef std::tuple<
        embeddingSegmentsSumParams,
        std::vector<InputShape>,  // input shapes
        ov::element::Type,        // embedding table
        ov::element::Type,        // indices
        std::string> embeddingSegmentsSumLayerTestParamsSet;

class EmbeddingSegmentsSumLayerTest : public testing::WithParamInterface<embeddingSegmentsSumLayerTestParamsSet>,
            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
