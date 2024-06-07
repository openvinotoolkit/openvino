// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
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
          │      ┌──────────────────┐    │
          └──────┤  DECONVOLUTION   ├────┘
                 └──┬───────────┬───┘
                    │           │
    ┌───────────────┴──┐     ┌──┴───────────────┐
    │     MULTIPLY     │     │     MULTIPLY     │
    └──────────────────┘     └──────────────────┘

Verify deconvolution node correctly handles
 multiple output edges on a single output port
 */

class DeconvMultipleOutputEdges : virtual public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        auto ngPrc = ov::element::f32;
        const ov::Shape inShape = {2, 12, 7, 7};
        const ov::Shape weiShape = {12, 6, 3, 3};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ngPrc, inShape),
                                        std::make_shared<ov::op::v0::Parameter>(ngPrc, weiShape)};

        auto deconv = utils::make_convolution_backprop_data(inputParams[0],
                                                            inputParams[1],
                                                            ov::element::f32,
                                                            ov::Strides({1, 1}),
                                                            ov::CoordinateDiff({0, 0}),
                                                            ov::CoordinateDiff({0, 0}),
                                                            ov::Strides({1, 1}),
                                                            ov::op::PadType::NOTSET,
                                                            false);
        deconv->get_rt_info() = CPUTestsBase::makeCPUInfo({nchw}, {nchw}, {});

        const auto const1 = ov::test::utils::make_constant(ngPrc, std::vector<size_t>{2, 6, 9, 9});
        const auto const2 = ov::test::utils::make_constant(ngPrc, std::vector<size_t>{2, 6, 9, 9});

        const auto mul1 = utils::make_eltwise(deconv->output(0), const1, utils::EltwiseTypes::MULTIPLY);
        const auto mul2 = utils::make_eltwise(deconv->output(0), const2, utils::EltwiseTypes::MULTIPLY);

        NodeVector results{mul1, mul2};
        function = std::make_shared<ov::Model>(results, inputParams, "DeconvMultipleOutputEdges");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

TEST_F(DeconvMultipleOutputEdges, smoke_DeconvMultipleOutputEdges_CPU) {
    run();
}

}  // namespace test
}  // namespace ov
