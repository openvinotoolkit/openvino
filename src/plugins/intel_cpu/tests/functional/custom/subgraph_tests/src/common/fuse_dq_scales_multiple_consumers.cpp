// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// FakeQuantize (u8) -> INT8 Convolution, ConvolutionBackpropData or MatMul -> Multiply by a per-channel constant,
// optionally with a second consumer of the same output. FuseConvMatmulFCDeconvAndDQScales used to fold a dequantization
// Multiply into the parent's DQ scales even with that consumer, scaling its output too. Snippets are disabled on static
// shapes, where they would take the Multiply.
enum class DQParent { Convolution, ConvolutionBackpropData };
enum class SecondConsumer { None, Result, Relu };

namespace {
std::string secondConsumerName(SecondConsumer secondConsumer) {
    switch (secondConsumer) {
    case SecondConsumer::None:
        return "None";
    case SecondConsumer::Result:
        return "Result";
    case SecondConsumer::Relu:
        return "Relu";
    }
    return "";
}

ResultVector makeResults(const std::shared_ptr<Node>& output, const Shape& scaleShape, SecondConsumer secondConsumer) {
    ResultVector results{std::make_shared<op::v0::Result>(
        std::make_shared<op::v1::Multiply>(output, utils::make_constant(element::f32, scaleShape)))};
    if (secondConsumer == SecondConsumer::Result) {
        results.push_back(std::make_shared<op::v0::Result>(output));
    } else if (secondConsumer == SecondConsumer::Relu) {
        results.push_back(std::make_shared<op::v0::Result>(std::make_shared<op::v0::Relu>(output)));
    }
    return results;
}
}  // namespace

using DQScalesConsumersParams = std::tuple<InputShape, DQParent, SecondConsumer>;

class FuseDQScalesConsumersCPUTest : public testing::WithParamInterface<DQScalesConsumersParams>,
                                     virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DQScalesConsumersParams>& obj) {
        const auto& [inputShape, parent, secondConsumer] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape
               << "_Parent=" << (parent == DQParent::Convolution ? "Convolution" : "ConvolutionBackpropData")
               << "_SecondConsumer=" << secondConsumerName(secondConsumer);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, parent, secondConsumer] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert(ov::hint::inference_precision(element::f32));
        init_input_shapes({inputShape});
        if (inputDynamicShapes[0].is_static()) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }

        const auto param = std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[0]);
        const auto fq = utils::make_fake_quantize(param, element::f32, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});

        // Per output channel: dimension 0 of Convolution weights, dimension 1 of ConvolutionBackpropData weights
        const bool isConvolution = parent == DQParent::Convolution;
        const auto weights = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v0::Convert>(
                utils::make_constant(element::i8, isConvolution ? Shape{8, 16, 1, 1} : Shape{16, 8, 1, 1}),
                element::f32),
            op::v0::Constant::create(
                element::f32,
                isConvolution ? Shape{8, 1, 1, 1} : Shape{1, 8, 1, 1},
                std::vector<float>{0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.008f, 0.009f}));
        std::shared_ptr<Node> output;
        if (isConvolution) {
            output = std::make_shared<op::v1::Convolution>(fq,
                                                           weights,
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1});
        } else {
            output = std::make_shared<op::v1::ConvolutionBackpropData>(fq,
                                                                       weights,
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1});
        }

        function = std::make_shared<Model>(makeResults(output, {1, 8, 1, 1}, secondConsumer),
                                           ParameterVector{param},
                                           "FuseDQScalesConsumers");
    }
};

TEST_P(FuseDQScalesConsumersCPUTest, CompareWithRefs) {
    run();
}

// Two FakeQuantized activations (u8 x i8) keep the MatMul INT8; constant weights would make it a FullyConnected.
// No single-consumer case: there the per-channel Multiply is fused as DQ scales, and the result is wrong on master
// independently of the consumer check (after a batch change).
using DQScalesMatMulConsumersParams = std::tuple<std::vector<InputShape>, SecondConsumer>;

class FuseDQScalesMatMulConsumersCPUTest : public testing::WithParamInterface<DQScalesMatMulConsumersParams>,
                                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DQScalesMatMulConsumersParams>& obj) {
        const auto& [inputShapes, secondConsumer] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShapes[0] << "_" << inputShapes[1]
               << "_SecondConsumer=" << secondConsumerName(secondConsumer);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShapes, secondConsumer] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert(ov::hint::inference_precision(element::f32));
        init_input_shapes(inputShapes);
        if (inputDynamicShapes[0].is_static()) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }

        ParameterVector params{std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[0]),
                               std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[1])};
        const auto fqA = utils::make_fake_quantize(params[0], element::f32, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        const auto fqB =
            utils::make_fake_quantize(params[1], element::f32, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        const auto matMul = std::make_shared<op::v0::MatMul>(fqA, fqB, false, false);
        function = std::make_shared<Model>(makeResults(matMul, {1, 1, 8}, secondConsumer),
                                           params,
                                           "FuseDQScalesMatMulConsumers");
    }

    // Small values keep each pair of u8 x i8 products within int16, which CPUs without VNNI saturate
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        inputs.insert({params[0],
                       utils::create_and_fill_tensor(element::f32,
                                                     targetInputStaticShapes[0],
                                                     utils::InputGenerateData(0, 1, 100))});
        inputs.insert({params[1],
                       utils::create_and_fill_tensor(element::f32,
                                                     targetInputStaticShapes[1],
                                                     utils::InputGenerateData(-1, 2, 100))});
    }
};

TEST_P(FuseDQScalesMatMulConsumersCPUTest, CompareWithRefs) {
    run();
}

namespace {
const std::vector<InputShape> inputShapes = {
    {{}, {{1, 16, 8, 8}}},
    {{-1, 16, -1, -1}, {{1, 16, 8, 8}, {2, 16, 5, 7}, {1, 16, 8, 8}}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_FuseDQScalesConsumers,
    FuseDQScalesConsumersCPUTest,
    ::testing::Combine(::testing::ValuesIn(inputShapes),
                       ::testing::Values(DQParent::Convolution, DQParent::ConvolutionBackpropData),
                       ::testing::Values(SecondConsumer::None, SecondConsumer::Result, SecondConsumer::Relu)),
    FuseDQScalesConsumersCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> matMulInputShapes = {
    {{{}, {{1, 4, 16}}}, {{}, {{1, 16, 8}}}},
    {{{-1, -1, 16}, {{1, 4, 16}, {2, 3, 16}, {1, 4, 16}}}, {{-1, 16, 8}, {{1, 16, 8}, {2, 16, 8}, {1, 16, 8}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_FuseDQScalesConsumersMatMul,
                         FuseDQScalesMatMulConsumersCPUTest,
                         ::testing::Combine(::testing::ValuesIn(matMulInputShapes),
                                            ::testing::Values(SecondConsumer::Result, SecondConsumer::Relu)),
                         FuseDQScalesMatMulConsumersCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
