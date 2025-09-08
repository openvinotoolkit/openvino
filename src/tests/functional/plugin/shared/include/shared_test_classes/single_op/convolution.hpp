// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
// ! [test_convolution:definition]
typedef std::tuple<
        std::vector<size_t>,    // Kernel size
        std::vector<size_t>,    // Strides
        std::vector<ptrdiff_t>, // Pad begin
        std::vector<ptrdiff_t>, // Pad end
        std::vector<size_t>,    // Dilation
        size_t,                 // Num out channels
        ov::op::PadType         // Padding type
> convSpecificParams;
typedef std::tuple<
        convSpecificParams,
        ov::element::Type,        // Model type
        std::vector<InputShape>,  // Input shapes
        std::string               // Device name
> convLayerTestParamsSet;

class ConvolutionLayerTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
// ! [test_convolution:definition]
}  // namespace test
}  // namespace ov
