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
using  groupConvBackpropSpecificParams = std::tuple<
    std::vector<size_t>,        // kernels
    std::vector<size_t>,        // strides
    std::vector<ptrdiff_t>,     // pad begins
    std::vector<ptrdiff_t>,     // pad ends
    std::vector<size_t>,        // dilations
    size_t,                     // num output channels
    size_t,                     // num groups
    ov::op::PadType,            // padding type
    std::vector<ptrdiff_t>>;    // output padding

using  groupConvBackpropLayerTestParamsSet = std::tuple<
    groupConvBackpropSpecificParams,
    ov::element::Type,        // Model type
    std::vector<InputShape>,  // Input shape
    ov::Shape,                // Output shapes
    std::string>;             // Device name

class GroupConvBackpropLayerTest : public testing::WithParamInterface<groupConvBackpropLayerTestParamsSet>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvBackpropLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
