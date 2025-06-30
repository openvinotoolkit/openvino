// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, can't be removed currently due to arm and kmb-plugin dependency (#55568)

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Strides
        std::vector<ptrdiff_t>,         // Pad begin
        std::vector<ptrdiff_t>,         // Pad end
        std::vector<size_t>,            // Dilation
        size_t,                         // Num out channels
        ov::op::PadType,                // Padding type
        std::vector<ptrdiff_t>          // Output padding
> convBackpropDataSpecificParams;
typedef std::tuple<
        convBackpropDataSpecificParams,
        ov::element::Type,              // Net precision
        std::vector<InputShape>,        // Input shapes
        ov::Shape,                      // Output shapes
        std::string                     // Device name
> convBackpropDataLayerTestParamsSet;

class ConvolutionBackpropDataLayerTest : public testing::WithParamInterface<convBackpropDataLayerTestParamsSet>,
                                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convBackpropDataLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
