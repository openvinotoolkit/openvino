// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
class ConvAndFQ : virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        const auto precision = element::f32;
        const auto input_static_shape = Shape{1, 3, 64, 64};

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
        // Weights
        auto weights_shape = Shape{1, 3, 64, 64};
        auto weights = utils::make_constant(element::i8, weights_shape, utils::InputGenerateData(-1, 2, 32768));
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto multiply = std::make_shared<op::v1::Multiply>(convert, op::v0::Constant::create(element::f32, {1, 1}, {0.625}));

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {1, 1};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 16;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = ov::test::utils::make_convolution(fq_before,
                                                     multiply,
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
        function = makeNgraphFunction(precision, input_params, fq_after, "ConvFQ");
    }
};

namespace {
TEST_F(ConvAndFQ, smoke_ConvAndFQ_CPU) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}
}  // namespace
}  // namespace test
}  // namespace ov
