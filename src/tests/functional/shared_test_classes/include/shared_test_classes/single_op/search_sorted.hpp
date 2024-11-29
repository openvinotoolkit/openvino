// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using SearchSortedSpecificParams = std::tuple<InputShape,  // sorted shape
                                              InputShape,  // values shape
                                              bool>;

using SearchSortedLayerTestParams = std::tuple<SearchSortedSpecificParams, ElementType, std::string>;

class SearchSortedLayerTest : public testing::WithParamInterface<SearchSortedLayerTestParams>,
                              public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SearchSortedLayerTestParams>& obj);
    static const std::vector<SearchSortedSpecificParams> GenerateParams();

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
