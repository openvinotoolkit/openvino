// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

class NotFusedConvSimpleOp : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector inputParams{
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 9}),
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 12, 9})};

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 16;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = ov::test::utils::make_convolution(inputParams[0],
                                                     element::f32,
                                                     kernelSize,
                                                     strides,
                                                     padBegin,
                                                     padEnd,
                                                     dilation,
                                                     paddingType,
                                                     numOutChannels);
        }
        const auto sharedNode = ov::test::utils::make_constant(element::f32, {1, 16, 1, 1});
        const auto postOpCandidate = ov::test::utils::make_eltwise(conv, sharedNode, utils::EltwiseTypes::ADD);
        const auto secondConsumpt = ov::test::utils::make_eltwise(inputParams[1], sharedNode, utils::EltwiseTypes::ADD);

        NodeVector results{postOpCandidate, secondConsumpt};
        function = std::make_shared<ov::Model>(results, inputParams, "NotFusedConvSimpleOp");
    }
};

TEST_F(NotFusedConvSimpleOp, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
