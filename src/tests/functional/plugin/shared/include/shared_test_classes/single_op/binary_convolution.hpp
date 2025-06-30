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

using binConvSpecificParams = std::tuple<
    std::vector<size_t>,            // Kernel size
    std::vector<size_t>,            // Strides
    std::vector<ptrdiff_t>,         // Pads begin
    std::vector<ptrdiff_t>,         // Pads end
    std::vector<size_t>,            // Dilations
    size_t,                         // Num Output channels
    ov::op::PadType,                // Padding type
    float>;                         // Padding value

using binaryConvolutionTestParamsSet = std::tuple<
    binConvSpecificParams,          //
    ov::element::Type,              // Model Type
    std::vector<InputShape>,        // Input shape
    std::string>;                   // Device name

class BinaryConvolutionLayerTest : public testing::WithParamInterface<binaryConvolutionTestParamsSet>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<binaryConvolutionTestParamsSet>& obj);
protected:
    void SetUp() override;
};

} // namespace test
} // namespace ov
