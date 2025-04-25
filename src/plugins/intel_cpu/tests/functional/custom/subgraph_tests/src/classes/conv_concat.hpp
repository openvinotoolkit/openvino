// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

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

using convConcatCPUParams = std::tuple<nodeType,                         // Node convolution type
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

namespace ConvConcat {
const std::vector<CPUSpecificParams> blockedCPUParams2D();
const std::vector<CPUSpecificParams> blockedCPUParams3D();

const ov::Shape inputShapes2D();
const ov::Shape inputShapes3D();
const int axis();
const ov::op::PadType paddingType();
const size_t numOutChannels();

const ov::Shape kernelSize3D();
const ov::Shape strides3D();
const std::vector<ptrdiff_t> padBegin3D();
const std::vector<ptrdiff_t> padEnd3D();
const ov::Shape dilation3D();

const ov::Shape kernelSize2D();
const ov::Shape strides2D();
const std::vector<ptrdiff_t> padBegin2D();
const std::vector<ptrdiff_t> padEnd2D();
const ov::Shape dilation2D();

const commonConvParams convParams2D();
const commonConvParams convParams3D();
const commonConvParams groupConvParams2D();
const commonConvParams groupConvParams3D();
} // namespace ConvConcat

}  // namespace test
}  // namespace ov
