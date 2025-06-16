// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*
┌──────────────────┐           ┌──────────────────┐
│      INPUT       │           │      WEIGHTS     │
└─────────┬────────┘           └─────────┬────────┘
          │                              │
┌─────────┴────────┐                     │
│  REORDER (NHWC)  │         NO REORDER  X
└─────────┬────────┘                     │
          │      ┌──────────────────┐    │
          └──────┤  DECONVOLUTION   ├────┘
                 └─────────┬────────┘
                           │
                 ┌─────────┴────────┐
                 │  REORDER (NCHW)  │
                 └─────────┬────────┘
                           │
                 ┌─────────┴────────┐
                 │      OUTPUT      │
                 └──────────────────┘
 */

class ReorderDeconvNHWCTest : virtual public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        const ov::Shape inShape = {2, 12, 7, 7};
        const ov::Shape weiShape = {12, 6, 3, 3};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inShape),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, weiShape)};

        auto deconv = utils::make_convolution_backprop_data(inputParams[0],
                                                            inputParams[1],
                                                            ov::element::f32,
                                                            ov::Strides({1, 1}),
                                                            ov::CoordinateDiff({0, 0}),
                                                            ov::CoordinateDiff({0, 0}),
                                                            ov::Strides({1, 1}),
                                                            ov::op::PadType::NOTSET,
                                                            false);
        deconv->get_rt_info() = CPUTestsBase::makeCPUInfo({nhwc}, {nhwc}, {});

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(deconv)};
        function = std::make_shared<ov::Model>(results, inputParams, "ReorderDeconvNHWC");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

TEST_F(ReorderDeconvNHWCTest, smoke_ReorderDeconvNHWC_CPU) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 2);
}

}  // namespace test
}  // namespace ov
