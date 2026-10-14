// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using PoolingFQTestParams = std::tuple<element::Type,  // Quantized precision
                                       size_t,         // Stride
                                       size_t,         // Padding
                                       bool,           // Exclude padding
                                       element::Type,  // Inference precision hint
                                       CPUSpecificParams>;

class PoolingAndFQ : public testing::WithParamInterface<PoolingFQTestParams>,
                     virtual public SubgraphBaseStaticTest,
                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PoolingFQTestParams>& obj) {
        const auto& [quantizedPrecision, stride, padding, excludePad, inferencePrecision, cpuParams] = obj.param;
        std::ostringstream result;
        result << "QuantizedPRC=" << quantizedPrecision << "_Stride=" << stride << "_Padding=" << padding
               << "_ExcludePad=" << excludePad << "_InferencePRC=" << inferencePrecision;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [quantizedPrecision, stride, padding, excludePad, inferencePrecision, cpuParams] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration = {{ov::hint::inference_precision(inferencePrecision)},
                         {ov::num_streams(1)},
                         {ov::inference_num_threads(1)}};
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        // CPU execution graphs report I8 primitives for both i8 and u8 tensors.
        selectedType = makeSelectedTypeStr(selectedType, element::i8);

        const bool signedDomain = quantizedPrecision == element::i8;
        const float outputLow = signedDomain ? -128.0f : 0.0f;
        const float outputHigh = signedDomain ? 127.0f : 255.0f;
        // The signed input shift is 127.75 and the output shift is -128.
        // Folding them before integer conversion gives the correct zero point.
        const float inputLow = signedDomain ? -255.5f : 0.5f;
        const float inputHigh = signedDomain ? 254.5f : 510.5f;
        const ov::Shape inputShape{1, 3, 9, 9};
        ov::ParameterVector params{std::make_shared<op::v0::Parameter>(element::f32, inputShape)};
        const auto inputFQ = utils::make_fake_quantize(params[0],
                                                       element::f32,
                                                       256,
                                                       {},
                                                       {outputLow},
                                                       {outputHigh},
                                                       {outputLow},
                                                       {outputHigh});
        const auto pooling = std::make_shared<op::v1::AvgPool>(inputFQ,
                                                               Strides{stride, stride},
                                                               Shape{padding, padding},
                                                               Shape{padding, padding},
                                                               Shape{2, 2},
                                                               excludePad,
                                                               op::RoundingType::FLOOR,
                                                               op::PadType::EXPLICIT);
        pooling->get_rt_info() = getCPUInfo();
        const auto outputFQ = utils::make_fake_quantize(pooling,
                                                        element::f32,
                                                        256,
                                                        {},
                                                        {inputLow},
                                                        {inputHigh},
                                                        {outputLow},
                                                        {outputHigh});

        // A quantized consumer keeps the output FQ fused into integer Pooling.
        // Identity weights keep each channel observable without cancellation.
        const auto order = op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});
        const auto transpose = std::make_shared<op::v1::Transpose>(outputFQ, order);
        const auto weights = op::v0::Constant::create(element::i8, Shape{3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
        const auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        const auto scale = op::v0::Constant::create(element::f32, Shape{}, {weightScale});
        const auto multiply = std::make_shared<op::v1::Multiply>(convert, scale);
        const auto matmul = std::make_shared<op::v0::MatMul>(transpose, multiply, false, false);
        function = std::make_shared<Model>(OutputVector{matmul}, params, "PoolingFQ");

        // Integer pooling can round before requantization; allow one output quantum.
        abs_threshold = weightScale + 1e-5f;
        rel_threshold = 0;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& param = function->get_parameters().front();
        const bool signedDomain = std::get<0>(GetParam()) == element::i8;
        const float outputLow = signedDomain ? -128.0f : 0.0f;
        const float outputHigh = signedDomain ? 127.0f : 255.0f;
        utils::InputGenerateData inputData;
        inputData.start_from = outputLow - 32.0f;
        inputData.range = 320;
        inputData.resolution = 4;
        auto tensor = utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[0], inputData);
        auto* data = tensor.data<float>();
        data[0] = 0.0f;
        data[1] = outputLow - 32.0f;
        data[2] = outputHigh + 32.0f;
        inputs.insert({param, tensor});
    }

    static constexpr float weightScale = 0.125f;
};

TEST_P(PoolingAndFQ, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Pooling", 1);
    CheckPluginRelatedResults(compiledModel, "Pooling");
    const auto expectedPrecision = std::get<0>(GetParam());
    const auto poolingNodes = GetNodesWithTypes(compiledModel.get_runtime_model(), {"Pooling"});
    for (const auto& node : poolingNodes) {
        const auto& rtInfo = node->get_rt_info();
        EXPECT_EQ(rtInfo.at(ov::exec_model_info::RUNTIME_PRECISION).as<element::Type>(), expectedPrecision);
        EXPECT_EQ(node->get_output_element_type(0), expectedPrecision);
    }
}

namespace {

const std::vector<CPUSpecificParams> cpuParams{
    {{nchw}, {nchw}, {}, "acl"},
    {{nhwc}, {nhwc}, {}, "acl"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PoolingAndFQ_NoPadding,
                         PoolingAndFQ,
                         ::testing::Combine(::testing::Values(element::i8, element::u8),
                                            ::testing::Values(1U, 2U),
                                            ::testing::Values(0U),
                                            ::testing::Values(false, true),
                                            ::testing::Values(element::f16, element::f32),
                                            ::testing::ValuesIn(cpuParams)),
                         PoolingAndFQ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PoolingAndFQ_ExcludePadding,
                         PoolingAndFQ,
                         ::testing::Combine(::testing::Values(element::i8, element::u8),
                                            ::testing::Values(1U, 2U),
                                            ::testing::Values(1U),
                                            ::testing::Values(true),
                                            ::testing::Values(element::f16, element::f32),
                                            ::testing::ValuesIn(cpuParams)),
                         PoolingAndFQ::getTestCaseName);

// ACL quantized NHWC pooling supports padding only when it is excluded from the average.
INSTANTIATE_TEST_SUITE_P(smoke_PoolingAndFQ_IncludePadding,
                         PoolingAndFQ,
                         ::testing::Combine(::testing::Values(element::i8, element::u8),
                                            ::testing::Values(1U, 2U),
                                            ::testing::Values(1U),
                                            ::testing::Values(false),
                                            ::testing::Values(element::f16, element::f32),
                                            ::testing::Values(cpuParams.front())),
                         PoolingAndFQ::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
