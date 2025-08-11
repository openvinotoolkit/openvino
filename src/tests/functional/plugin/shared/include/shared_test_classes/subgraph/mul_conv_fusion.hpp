// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::NodeTypeInfo,   // Convolution type
                   ov::Shape,          // Input shape
                   ov::Shape,          // Weights shape
                   ov::Shape,          // Const shape
                   ov::element::Type,  // Network precision
                   bool,               // True if test is negative
                   std::string         // Device name
                   >
    MulConvFusionParams;

class MulConvFusion : public testing::WithParamInterface<MulConvFusionParams>,
                      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MulConvFusionParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
