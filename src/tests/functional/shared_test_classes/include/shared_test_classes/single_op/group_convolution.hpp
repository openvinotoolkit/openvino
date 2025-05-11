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
        std::vector<size_t>,
        std::vector<size_t>,
        std::vector<ptrdiff_t>,
        std::vector<ptrdiff_t>,
        std::vector<size_t>,
        size_t,
        size_t,
        ov::op::PadType> groupConvSpecificParams;
typedef std::tuple<
        groupConvSpecificParams,
        ov::element::Type,
        std::vector<InputShape>,
        std::string> groupConvLayerTestParamsSet;

class GroupConvolutionLayerTest : public testing::WithParamInterface<groupConvLayerTestParamsSet>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
