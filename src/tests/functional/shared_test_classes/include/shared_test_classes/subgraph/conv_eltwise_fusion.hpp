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

typedef std::tuple<std::tuple<ov::NodeTypeInfo,  // Convolution type
                              size_t             // Number of inputs
                              >,
                   ov::NodeTypeInfo,   // Eltwise type
                   bool,               // Is the test negative or not
                   ov::Shape,          // Input shape
                   ov::Shape,          // Weights shape
                   ov::Shape,          // Const shape
                   ov::element::Type,  // Network precision
                   std::string         // Device name
                   >
    ConvEltwiseFusionParams;

class ConvEltwiseFusion : public testing::WithParamInterface<ConvEltwiseFusionParams>,
                          virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
