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
        std::vector<size_t>,    // Strides
        std::vector<ptrdiff_t>, // Pad begin
        std::vector<ptrdiff_t>, // Pad end
        std::vector<size_t>,    // Dilation
        size_t,                 // Groups
        size_t,                 // Deformable groups
        size_t,                 // Num out channels
        ov::op::PadType,        // Padding type
        bool                   // Bilinear interpolation pad
> deformableConvSpecificParams;
typedef std::tuple<
        deformableConvSpecificParams,
        bool,                      // Modulation
        ov::element::Type,         // Model type
        std::vector<InputShape>,   // Input shapes
        std::string                // Device name
> deformableConvLayerTestParamsSet;

class DeformableConvolutionLayerTest : public testing::WithParamInterface<deformableConvLayerTestParamsSet>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<deformableConvLayerTestParamsSet>& obj);
protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
