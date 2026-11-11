// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/single_op/convolution.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/type_relaxed.hpp"

#include <algorithm>

namespace {
using ov::test::InputShape;
using ov::test::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ov::element::Type,     // Net precision
        ov::element::Type,     // Input precision
        ov::element::Type,     // Output precision
        InputShape,            // Input shape
        bool,                  // ReduceSum replacement test
        std::string            // Device name
> convLayerTestParamsSet;

class ConvolutionLayerGPUTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
        const auto& [convParams, netType, inType, outType, inputShape, is_ReduceSum_test, targetDevice] = obj.param;

        const auto& [kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType] = convParams;

        std::ostringstream result;
        result << "IS=";
        result  << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        if (is_ReduceSum_test)
            result << "K" << ov::test::utils::vec2str(std::vector<size_t>{inputShape.second[0].at(-2), inputShape.second[0].at(-1)}) << "_";
        else
            result << "K" << ov::test::utils::vec2str(kernel) << "_";
        result << "S" << ov::test::utils::vec2str(stride) << "_";
        result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "is_ReduceSum_test=" << is_ReduceSum_test << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [convParams, netType, _inType, _outType, inputShape, is_ReduceSum_test, _targetDevice] = this->GetParam();
        inType = _inType;
        outType = _outType;
        targetDevice = _targetDevice;

        init_input_shapes({inputShape});

        const auto& [_kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType] = convParams;
        auto kernel = _kernel;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        if (is_ReduceSum_test) {
            auto input_static_shape = inputDynamicShapes[0].to_shape();
            kernel.clear();
            kernel.push_back(input_static_shape.at(-2));
            kernel.push_back(input_static_shape.at(-1));
        }
        auto convolutionNode = ov::test::utils::make_convolution(inputParams.front(), netType, kernel, stride, padBegin,
                                                                 padEnd, dilation, padType, convOutChannels);

        ov::ResultVector results;
        for (size_t i = 0; i < convolutionNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(convolutionNode->output(i)));

        function = std::make_shared<ov::Model>(results, inputParams, "Convolution");
    }
};

TEST_P(ConvolutionLayerGPUTest, Inference) {
    run();
}

// Check 3D input tensor for convolution is handled properly and its output is correct comparing with ov runtime.
INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_3D_tensor_basic,
                         ConvolutionLayerGPUTest,
                         ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>{3}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0}),
                                                               ::testing::Values(std::vector<size_t>{1}),
                                                               ::testing::Values(13),
                                                               ::testing::Values(ov::op::PadType::SAME_UPPER)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(InputShape{{}, {{1, 13, 30}}}),
                                            ::testing::Values(false),
                                            ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerGPUTest::getTestCaseName);

// Customer model input/filter shape
std::vector<InputShape> input_shape_reducesum_test = {
    {{}, {{1, 1, 36, 64}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionLayerGPUTest_4D_tensor_ReduceSum,
                         ConvolutionLayerGPUTest,
                         ::testing::Combine(::testing::Combine(::testing::Values(std::vector<size_t>{36, 64}),
                                                               ::testing::Values(std::vector<size_t>{1, 1}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                                                               ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
                                                               ::testing::Values(std::vector<size_t>{1, 1}),
                                                               ::testing::Values(1),
                                                               ::testing::Values(ov::op::PadType::EXPLICIT)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(input_shape_reducesum_test),
                                            ::testing::Values(true),
                                            ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerGPUTest::getTestCaseName);
}  // namespace

namespace ov::test::intel_gpu {

class ConvolutionDeclaredOutputTypeTest : public testing::Test {
protected:
    ov::AnyMap config() const {
        return {ov::hint::inference_precision(ov::element::dynamic),
                ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
                ov::num_streams(1)};
    }
};

TEST_F(ConvolutionDeclaredOutputTypeTest, F16InputsRetainExactF32Product) {
    const ov::Shape shape{1, 1, 1, 1};
    constexpr float operand = 1.0009765625f;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
    auto weights = ov::op::v0::Constant::create(ov::element::f16, shape, {operand});
    auto base = std::make_shared<ov::op::v1::Convolution>(
        input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
    auto convolution = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Convolution>>(
        *base, ov::element::TypeVector{ov::element::f32, ov::element::f32},
        ov::element::TypeVector{ov::element::f32});
    convolution->set_friendly_name("conv");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{convolution->output(0)}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU", config());
    auto request = compiled.create_infer_request();
    request.get_input_tensor().data<ov::float16>()[0] = ov::float16(operand);
    request.infer();
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_element_type(), ov::element::f32);
    ASSERT_EQ(output.get_shape(), shape);
    // Both operands are exactly representable in F16; rounding their product
    // to F16 before the public F32 output would lose the final 2^-20 term.
    ASSERT_EQ(output.data<const float>()[0], operand * operand);
}

TEST_F(ConvolutionDeclaredOutputTypeTest, I32FallbackRetainsPublicOutputType) {
    const ov::Shape shape{1, 1, 2, 2};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape);
    auto weights = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 1, 1, 1}, {3});
    auto convolution = std::make_shared<ov::op::v1::Convolution>(
        input, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
    convolution->set_friendly_name("conv");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{convolution->output(0)}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU", config());
    auto request = compiled.create_infer_request();
    const std::vector<int32_t> values{2, -3, 4, -5};
    std::copy(values.begin(), values.end(), request.get_input_tensor().data<int32_t>());
    request.infer();
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_element_type(), ov::element::i32);
    ASSERT_EQ(output.get_shape(), shape);
    for (size_t idx = 0; idx < values.size(); ++idx) {
        ASSERT_EQ(output.data<const int32_t>()[idx], values[idx] * 3) << "index " << idx;
    }
}

}  // namespace ov::test::intel_gpu
