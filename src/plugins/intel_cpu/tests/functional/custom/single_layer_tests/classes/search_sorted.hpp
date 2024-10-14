// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_tensor_utils.hpp"
#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SearchSorted {
using SearchSortedSpecificParams = std::tuple<InputShape,  // sorted shape
                                              InputShape,  // values shape
                                              bool>;

using SearchSortedLayerTestParams = std::tuple<SearchSortedSpecificParams, ElementType>;

using SearchSortedLayerCPUTestParamsSet = SearchSortedLayerTestParams;

class SearchSortedLayerCPUTest : public testing::WithParamInterface<SearchSortedLayerCPUTestParamsSet>,
                                 public SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SearchSortedLayerCPUTestParamsSet> obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

const std::vector<ElementType> indexPrecisions = {ElementType::i32, ElementType::i64};

extern const std::vector<SearchSortedSpecificParams> SearchSortedParamsVector;
}  // namespace SearchSorted
}  // namespace test
}  // namespace ov
