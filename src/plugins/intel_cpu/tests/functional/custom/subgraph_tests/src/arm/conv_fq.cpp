// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/util/common_util.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<
        InputShape,                        // input shape
        element::Type,                     // input precision
        std::vector<std::vector<float>>,   // quantize intervals
        std::vector<size_t>,               // fq const shapes
        bool,                              // bias presence
        std::string                        // device name
> ConvAndFQTestParams;

class ConvAndFQ : public testing::WithParamInterface<ConvAndFQTestParams>,
                  virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvAndFQTestParams>& obj) {
        const auto& [inputShape, inputPrecision, quantizeIntervals, fqConstShapes, withBias, targetName] = obj.param;
        std::ostringstream results;

        results << "IS=" << inputShape << "_InPRC=" << inputPrecision
                << "_Intervals=";
        for (const auto& vecInt : quantizeIntervals) {
            results << ov::util::vector_to_string(vecInt) << ",";
        }
        results << "_fqShapes=" << ov::util::vector_to_string(fqConstShapes)
                << "_withBias=" << withBias
                << "_targetDevice=" << targetName;

        return results.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, inputPrecision, quantizeIntervals, fqConstShapes, withBias, targetName] = this->GetParam();
        abs_threshold = 4e-3f;
        targetDevice = targetName;
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        init_input_shapes({inputShape});
        ov::ParameterVector input_params{
            std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0])};

        auto fq_before = ov::test::utils::make_fake_quantize(input_params[0],
                                                             inputPrecision,
                                                             256,
                                                             fqConstShapes,
                                                             quantizeIntervals[0],
                                                             quantizeIntervals[1],
                                                             quantizeIntervals[2],
                                                             quantizeIntervals[3]);

        auto weights = utils::make_constant(element::i8, {4, 3, 2, 2});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto multiply = std::make_shared<op::v1::Multiply>(convert, op::v0::Constant::create(element::f32, {4, 1, 1, 1}, {0.625}));

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {1, 1};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 4;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = ov::test::utils::make_convolution(fq_before,
                                                     multiply,
                                                     inputPrecision,
                                                     kernelSize,
                                                     strides,
                                                     padBegin,
                                                     padEnd,
                                                     dilation,
                                                     paddingType,
                                                     numOutChannels);
        }

        auto fqInput = conv;
        if (withBias) {
            auto bias = ov::test::utils::make_constant(ov::element::f16, ov::Shape({1, 4, 1, 1}));
            auto convertBias = std::make_shared<op::v0::Convert>(bias, element::f32);
            auto convBiasAdd = std::make_shared<ov::op::v1::Add>(conv, convertBias);
            fqInput = convBiasAdd;
        }

        auto fq_after = ov::test::utils::make_fake_quantize(fqInput,
                                                            inputPrecision,
                                                            256,
                                                            {},
                                                            {quantizeIntervals[0][0]},
                                                            {quantizeIntervals[1][0]},
                                                            {quantizeIntervals[2][0]},
                                                            {quantizeIntervals[3][0]});

        std::shared_ptr<Node> pooling;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {2, 2};
            const std::vector<size_t> padBegin = {1, 1};
            const std::vector<size_t> padEnd = {1, 1};
            const op::PadType paddingType = op::PadType::EXPLICIT;
            ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;
            pooling = std::make_shared<ov::op::v1::MaxPool>(fq_after,
                                                            strides,
                                                            padBegin,
                                                            padEnd,
                                                            kernelSize,
                                                            roundingType,
                                                            paddingType);
        }

        function = makeNgraphFunction(inputPrecision, input_params, pooling, "ConvFQ");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
            const auto& funcInput = funcInputs[0];
            ov::Tensor tensor;
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 256;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    void checkConvolutionPrecision(ov::element::Type expectedPrecision) {
        const auto runtime_model = compiledModel.get_runtime_model();
        for (auto& op : runtime_model->get_ops()) {
            if (op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "Convolution") {
                EXPECT_EQ(op->get_rt_info().at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>(), expectedPrecision);
            }
        }
    }
};

TEST_P(ConvAndFQ, CompareWithRefs) {
    run();

    // per channel dequantization for quantized convolution is not supported by ACL executor,
    // so in this case we fallback to f32 implementation
    ov::element::Type expectedPrecision = element::f32;
#if defined(OPENVINO_ARCH_ARM64)
    const auto& [inputShape, inputPrecision, quantizeIntervals, fqConstShapes, withBias, targetName] = this->GetParam();
    if (fqConstShapes.empty()) {
        expectedPrecision = quantizeIntervals[0][0] < 0.f ? element::i8 : element::u8;
    }
#endif

    checkConvolutionPrecision(expectedPrecision);
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {

std::vector<InputShape> inputShapes{{{}, {{1, 3, 2, 2}}},
                                   {{-1, 3, -1, 2}, {{1, 3, 4, 2}}}};

std::vector<std::vector<std::vector<float>>> perTensorQuantizeIntervals{
    {{-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
    {{0.f}, {2.55f}, {0.f}, {2.55f}},
};

std::vector<std::vector<std::vector<float>>> perChannelQuantizeIntervals{
    {{-1.28f, -1.0f, -0.8f}, {1.27f, 1.0f, 0.8f}, {-1.28f, -1.0f, -0.8f}, {1.27f, 1.0f, 0.8f}},
};

std::vector<bool> withBias{false, true};

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQ_CPU_PerTensor,
                         ConvAndFQ,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(perTensorQuantizeIntervals),
                                            ::testing::Values(std::vector<size_t>{}),
                                            ::testing::ValuesIn(withBias),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQ_CPU_PerChannel,
                         ConvAndFQ,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(perChannelQuantizeIntervals),
                                            ::testing::Values(std::vector<size_t>{1, 3, 1, 1}),
                                            ::testing::ValuesIn(withBias),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov