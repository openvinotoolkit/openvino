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
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// INT8 MatMul whose per-output-channel scales are passed to oneDNN as weight scales. Their mask has to name the last
// weights dimension (N), plus the batch dimensions for scales that also vary per batch. Snippets are disabled on static
// shapes, where they would take the Multiply.

class MatMulInt8ScalesTestBase : virtual public SubgraphBaseTest {
protected:
    void configure() {
        targetDevice = utils::DEVICE_CPU;
        // Integer sums that cancel to 0 come out of the f32 reference as about 1e-6
        abs_threshold = 1e-4;
        configuration.insert(ov::hint::inference_precision(element::f32));
        if (inputDynamicShapes[0].is_static()) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }
    }

    // Values in [0, 1] and [-1, 1] keep each pair of u8 x i8 products within int16, which CPUs without VNNI saturate
    static ov::Tensor makeInput(const ov::Shape& shape, bool isSigned) {
        return utils::create_and_fill_tensor(element::f32,
                                             shape,
                                             utils::InputGenerateData(isSigned ? -1 : 0, isSigned ? 2 : 1, 100));
    }
};

// FakeQuantize (u8) x FakeQuantize (i8) -> MatMul -> Multiply by a per-output-channel constant, which is fused into
// the MatMul as dequantization scales
using MatMulInt8WeiScalesParams = std::tuple<std::vector<InputShape>, bool>;

class MatMulInt8WeiScalesCPUTest : public testing::WithParamInterface<MatMulInt8WeiScalesParams>,
                                   public MatMulInt8ScalesTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulInt8WeiScalesParams>& obj) {
        const auto& [inputShapes, transposeB] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShapes[0] << "_" << inputShapes[1] << "_transposeB=" << transposeB;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShapes, transposeB] = GetParam();
        init_input_shapes(inputShapes);
        configure();

        ParameterVector params{std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[0]),
                               std::make_shared<op::v0::Parameter>(element::f32, inputDynamicShapes[1])};
        const auto fqA = utils::make_fake_quantize(params[0], element::f32, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        const auto fqB =
            utils::make_fake_quantize(params[1], element::f32, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        const auto matMul = std::make_shared<op::v0::MatMul>(fqA, fqB, false, transposeB);

        const auto& shapeB = inputDynamicShapes[1];
        const auto rank = shapeB.size();
        Shape scaleShape(rank, 1);
        scaleShape.back() = static_cast<size_t>(shapeB[transposeB ? rank - 2 : rank - 1].get_length());
        const auto multiply =
            std::make_shared<op::v1::Multiply>(matMul, utils::make_constant(element::f32, scaleShape));
        function = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(multiply)},
                                           params,
                                           "MatMulInt8WeiScales");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        inputs.insert({params[0], makeInput(targetInputStaticShapes[0], false)});
        inputs.insert({params[1], makeInput(targetInputStaticShapes[1], true)});
    }
};

TEST_P(MatMulInt8WeiScalesCPUTest, CompareWithRefs) {
    run();
}

// FakeQuantize (u8) x batched int8 weights [B, K, N] dequantized by a Multiply, optionally followed by Relu and a
// Multiply by a per-output-channel constant, which is folded into the weight scales: a FullyConnected with batched
// weights on x86, a MatMul on ARM. Weight scales that vary only per batch ([B, 1, 1]) are not covered: the oneDNN
// kernels used on x86 read a mask with batch bits only as per-output-channel scales.
using MatMulInt8BatchedWeightsScalesParams = std::tuple<InputShape, Shape, bool>;

class MatMulInt8BatchedWeightsScalesCPUTest : public testing::WithParamInterface<MatMulInt8BatchedWeightsScalesParams>,
                                              public MatMulInt8ScalesTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulInt8BatchedWeightsScalesParams>& obj) {
        const auto& [inputShape, weightScaleShape, reluAndMultiply] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_weightScales=" << weightScaleShape << "_reluAndMultiply=" << reluAndMultiply;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, weightScaleShape, reluAndMultiply] = GetParam();
        init_input_shapes({inputShape});
        configure();

        const auto& shapeA = inputDynamicShapes[0];
        const auto batch = static_cast<size_t>(shapeA[0].get_length());
        const auto K = static_cast<size_t>(shapeA[2].get_length());
        const auto param = std::make_shared<op::v0::Parameter>(element::f32, shapeA);
        const auto fq = utils::make_fake_quantize(param, element::f32, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        const size_t N = 24;
        const auto weights = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v0::Convert>(utils::make_constant(element::i8, Shape{batch, K, N}), element::f32),
            utils::make_constant(element::f32, weightScaleShape));
        std::shared_ptr<Node> output = std::make_shared<op::v0::MatMul>(fq, weights, false, false);
        if (reluAndMultiply) {
            output = std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::Relu>(output),
                                                        utils::make_constant(element::f32, Shape{1, 1, N}));
        }
        function = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(output)},
                                           ParameterVector{param},
                                           "MatMulInt8BatchedWeightsScales");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs.insert({function->get_parameters()[0], makeInput(targetInputStaticShapes[0], false)});
    }
};

TEST_P(MatMulInt8BatchedWeightsScalesCPUTest, CompareWithRefs) {
#if defined(OPENVINO_ARCH_X86_64)
    // Without AVX-512 the oneDNN gemm kernel applies one row of scales that vary per batch and output channel
    const auto& weightScaleShape = std::get<1>(GetParam());
    if (weightScaleShape[0] > 1 && !ov::with_cpu_x86_avx512_core()) {
        GTEST_SKIP() << "scales per batch and output channel are not supported by the gemm kernel";
    }
#endif
    run();
}

namespace {

const std::vector<std::vector<InputShape>> matMulInputShapes = {
    {{{}, {{5, 16}}}, {{}, {{16, 24}}}},
    {{{}, {{2, 5, 16}}}, {{}, {{2, 16, 24}}}},
    {{{}, {{2, 5, 16}}}, {{}, {{1, 16, 24}}}},
    {{{}, {{1, 3, 5, 16}}}, {{}, {{1, 3, 16, 24}}}},
    {{{-1, -1, 16}, {{1, 5, 16}, {2, 3, 16}, {3, 5, 16}}}, {{-1, 16, 24}, {{1, 16, 24}, {2, 16, 24}, {3, 16, 24}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulInt8WeiScales,
                         MatMulInt8WeiScalesCPUTest,
                         ::testing::Combine(::testing::ValuesIn(matMulInputShapes), ::testing::Values(false)),
                         MatMulInt8WeiScalesCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> matMulTransposedInputShapes = {
    {{{}, {{2, 5, 16}}}, {{}, {{2, 24, 16}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulInt8WeiScalesTransposedB,
                         MatMulInt8WeiScalesCPUTest,
                         ::testing::Combine(::testing::ValuesIn(matMulTransposedInputShapes), ::testing::Values(true)),
                         MatMulInt8WeiScalesCPUTest::getTestCaseName);

const std::vector<InputShape> batchedWeightsInputShapes = {
    {{}, {{2, 5, 16}}},
    {{2, -1, 16}, {{2, 5, 16}, {2, 7, 16}, {2, 5, 16}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulInt8BatchedWeightsScales,
                         MatMulInt8BatchedWeightsScalesCPUTest,
                         ::testing::Combine(::testing::ValuesIn(batchedWeightsInputShapes),
                                            ::testing::Values(Shape{1, 1, 24}, Shape{2, 1, 24}),
                                            ::testing::Values(false)),
                         MatMulInt8BatchedWeightsScalesCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulInt8BatchedWeightsScalesReluAndMultiply,
                         MatMulInt8BatchedWeightsScalesCPUTest,
                         ::testing::Combine(::testing::ValuesIn(batchedWeightsInputShapes),
                                            ::testing::Values(Shape{1, 1, 1}),
                                            ::testing::Values(true)),
                         MatMulInt8BatchedWeightsScalesCPUTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
