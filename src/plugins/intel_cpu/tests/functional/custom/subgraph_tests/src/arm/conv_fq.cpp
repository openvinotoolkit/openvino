// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
class ConvAndFQ : virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        const auto precision = element::f32;
        const auto input_static_shape = Shape{16, 3, 9, 9};

        auto in_shapes = static_shapes_to_test_representation({input_static_shape});
        init_input_shapes({in_shapes});
        ov::ParameterVector input_params{
            std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape(input_static_shape))};

        auto fq_before = ov::test::utils::make_fake_quantize(input_params[0], precision, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

        // Weights
        auto weights_shape = Shape{16, 3, 9, 9};
        auto weights = utils::make_constant(element::i8, weights_shape, ov::test::utils::InputGenerateData(0, 2, 1));
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto multiply = std::make_shared<op::v1::Multiply>(convert, op::v0::Constant::create(element::f32, {1, 1}, {0.1}));

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {3, 3};
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

        auto add = std::make_shared<ov::op::v1::Add>(conv, op::v0::Constant::create(element::f32, {1, 1}, {0.625}));
        auto fq_after = ov::test::utils::make_fake_quantize(add, precision, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

        auto matmul_const = ov::test::utils::make_constant(ov::element::i8, {1, 1});
        auto convert_mm = std::make_shared<op::v0::Convert>(matmul_const, element::f32);
        auto multiply_mm = std::make_shared<op::v1::Multiply>(convert_mm, op::v0::Constant::create(element::f32, {1, 1}, {0.625}));
        const auto matMul = std::make_shared<ov::op::v0::MatMul>(fq_after, multiply_mm, false, false);

        function = makeNgraphFunction(precision, input_params, matMul, "ConvFQ");
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