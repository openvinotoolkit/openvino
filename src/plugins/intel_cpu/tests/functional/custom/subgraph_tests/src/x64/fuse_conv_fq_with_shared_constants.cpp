// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
class ConvAndFQWithSharedConstants : virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        fusedOps = std::vector<std::string>{"FakeQuantize"};
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        const auto precision = element::f32;
        const auto input_static_shape = Shape{1, 3, 40, 40};

        auto in_shapes = static_shapes_to_test_representation({input_static_shape});
        init_input_shapes({in_shapes});
        ov::ParameterVector input_params{
            std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape(input_static_shape))};

        auto shared_il = ov::op::v0::Constant::create(precision, {1, 1, 1, 1}, {0.f});
        auto shared_ih = ov::op::v0::Constant::create(precision, {1, 1, 1, 1}, {12.5f});
        auto shared_ol = ov::op::v0::Constant::create(precision, {1, 1, 1, 1}, {0.f});
        auto shared_oh = ov::op::v0::Constant::create(precision, {1, 1, 1, 1}, {12.5f});
        auto fq_before = std::make_shared<ov::op::v0::FakeQuantize>(input_params[0],
                                                                    shared_il,
                                                                    shared_ih,
                                                                    shared_ol,
                                                                    shared_oh,
                                                                    256);

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {2, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 16;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = ov::test::utils::make_convolution(fq_before,
                                                     precision,
                                                     kernelSize,
                                                     strides,
                                                     padBegin,
                                                     padEnd,
                                                     dilation,
                                                     paddingType,
                                                     numOutChannels);
        }

        auto fq_after =
            std::make_shared<ov::op::v0::FakeQuantize>(conv, shared_il, shared_ih, shared_ol, shared_oh, 256);
        function = makeNgraphFunction(precision, input_params, fq_after, "ConvFQWithSharedContants");
    }
};

namespace {
TEST_F(ConvAndFQWithSharedConstants, smoke_ConvAndFQWithSharedConstants_CPU) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}
}  // namespace
}  // namespace test
}  // namespace ov
