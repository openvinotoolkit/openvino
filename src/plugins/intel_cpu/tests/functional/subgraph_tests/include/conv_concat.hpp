// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using commonConvParams = std::tuple<std::vector<size_t>,     // Kernel size
                                    std::vector<size_t>,     // Strides
                                    std::vector<ptrdiff_t>,  // Pad begin
                                    std::vector<ptrdiff_t>,  // Pad end
                                    std::vector<size_t>,     // Dilation
                                    size_t,                  // Num out channels
                                    ov::op::PadType,         // Padding type
                                    size_t                   // Number of groups
                                    >;

using convConcatCPUParams = std::tuple<nodeType,                         // Ngraph convolution type
                                       commonConvParams,                 // Convolution params
                                       CPUTestUtils::CPUSpecificParams,  // CPU runtime params
                                       ov::Shape,                        // Input shapes
                                       int                               // Axis for concat
                                       >;

class ConvConcatSubgraphTest : public testing::WithParamInterface<convConcatCPUParams>,
                               public CPUTestsBase,
                               virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj);

protected:
    void SetUp() override;
    std::string pluginTypeNode;
};

}  // namespace test
}  // namespace ov
