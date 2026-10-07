// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
 * ACL low-precision FullyConnected with constant i8 weights and an i8 destination:
 *   FakeQuantize -> MatMul(i8, i8) -> Multiply -> Add(bias) -> FakeQuantize
 * ACLLowpFullyConnectedExecutor repacks weights into the arm_compute::WeightFormat ACL reports.
 * A non-concrete format (ANY/UNSPECIFIED) gives a zero-sized weights buffer and a SIGSEGV in
 * compile_model(). An f32 destination takes a different probe arm, so keep the i8 output.
 */

typedef std::tuple<InputShape,         // input shape
                   element::Type,      // network precision
                   CPUSpecificParams,  // expected implementation
                   std::string         // device name
                   >
    FCAndFQTestParams;

class FCAndFQ : public testing::WithParamInterface<FCAndFQTestParams>,
                virtual public SubgraphBaseTest,
                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FCAndFQTestParams>& obj) {
        const auto& [inputShape, netPrecision, cpuParams, targetName] = obj.param;
        std::ostringstream results;
        results << "IS=" << inputShape << "_netPRC=" << netPrecision << "_targetDevice=" << targetName
                << CPUTestsBase::getTestCaseName(cpuParams);
        return results.str();
    }

protected:
    static constexpr size_t inChannels = 64;
    static constexpr size_t outChannels = 32;

    void SetUp() override {
        const auto& [inputShape, netPrecision, cpuParams, targetName] = this->GetParam();
        abs_threshold = 1e-2f;
        targetDevice = targetName;
        // Pin the implementation: the whole point is that the ACL low-precision executor runs.
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        init_input_shapes({inputShape});

        ov::ParameterVector input_params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};

        // Signed activation range -> i8 activations.
        auto fq_before = ov::test::utils::make_fake_quantize(input_params[0],
                                                            netPrecision,
                                                            256,
                                                            {},
                                                            {-1.28f},
                                                            {1.27f},
                                                            {-1.28f},
                                                            {1.27f});

        // Constant i8 weights with the dequantization Convert+Multiply that LPT leaves behind.
        // Per-tensor only: ACL rejects per-channel dequantization (fullyconnected_implementations.cpp).
        auto weights = ov::test::utils::make_constant(element::i8, ov::Shape{outChannels, inChannels});
        auto convert_weights = std::make_shared<op::v0::Convert>(weights, netPrecision);
        auto dequantized_weights =
            std::make_shared<op::v1::Multiply>(convert_weights,
                                               op::v0::Constant::create(netPrecision, {1, 1}, {0.015f}));

        // transpose_b == true -> weights are [outChannels, inChannels], i.e. a FullyConnected.
        std::shared_ptr<ov::Node> fqInput =
            std::make_shared<ov::op::v0::MatMul>(fq_before, dequantized_weights, false, true);

        // Bias is mandatory: FCMulAddFQBlock matches MatMul -> Multiply -> Add -> FakeQuantize, so
        // without a non-i32 constant Add the output FakeQuantize is not fused and the dst stays f32.
        auto bias = ov::test::utils::make_constant(element::f16, ov::Shape{1, outChannels});
        auto convert_bias = std::make_shared<op::v0::Convert>(bias, netPrecision);
        fqInput = std::make_shared<ov::op::v1::Add>(fqInput, convert_bias);

        // Signed output range -> i8 destination. Do not relax to f32, see the note above.
        auto fq_after =
            ov::test::utils::make_fake_quantize(fqInput, netPrecision, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

        // Without a quantized consumer the output FakeQuantize is not fused and the FC dst stays
        // f32. Weights are tiny and uniform so a 1-LSB rounding difference is not amplified.
        auto tail_weights =
            op::v0::Constant::create(element::i8, ov::Shape{1, outChannels}, std::vector<int8_t>(outChannels, 1));
        auto tail_convert = std::make_shared<op::v0::Convert>(tail_weights, netPrecision);
        auto tail_dequantized =
            std::make_shared<op::v1::Multiply>(tail_convert,
                                               op::v0::Constant::create(netPrecision, {1, 1}, {0.01f}));
        auto tail = std::make_shared<ov::op::v0::MatMul>(fq_after, tail_dequantized, false, true);

        function = create_ov_model(netPrecision, input_params, tail, "FCAndFQ");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        // Model::inputs() returns by value; bind the vector, not an element of the temporary.
        const auto funcInputs = function->inputs();
        const auto& funcInput = funcInputs[0];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 256;
        auto tensor =
            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }

    // Fails if the graph stops being quantized or leaves ACL, so the test cannot pass vacuously.
    void checkQuantizedAclFullyConnected() {
        const auto runtime_model = compiledModel.get_runtime_model();
        size_t quantizedAclCount = 0;
        for (const auto& op : runtime_model->get_ops()) {
            const auto& rt = op->get_rt_info();
            if (rt.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() != "FullyConnected") {
                continue;
            }
            const auto runtimePrecision = rt.at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>();
            const auto outputPrecision = rt.at(ov::exec_model_info::OUTPUT_PRECISIONS).as<ov::element::Type>();
            const auto implType = rt.at(ov::exec_model_info::IMPL_TYPE).as<std::string>();
            if (runtimePrecision == element::i8 && outputPrecision == element::i8 &&
                implType.find("acl") != std::string::npos) {
                ++quantizedAclCount;
            }
        }
        EXPECT_EQ(quantizedAclCount, 1U)
            << "Expected one ACL FullyConnected with i8 activations and an i8 destination";
    }
};

TEST_P(FCAndFQ, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // This used to SIGSEGV in compile_model(), so completing run() is itself part of the check.
    run();

    checkQuantizedAclFullyConnected();
    CheckPluginRelatedResults(compiledModel, "FullyConnected");
}

// On 32-bit ARM filterCPUInfo() leaves no parameters, so the suite is legitimately empty there.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FCAndFQ);

namespace {

// 3D matches the [batch, tokens, hidden] shape of the transformer models that hit this.
std::vector<InputShape> inputShapes{
    {{}, {{16, 64}}},
    {{}, {{1, 16, 64}}},
    {{-1, -1, 64}, {{1, 16, 64}, {1, 32, 64}}},
};

const std::vector<CPUSpecificParams> cpuParams{
    CPUSpecificParams{{}, {}, {}, CPUTestsBase::makeSelectedTypeStr("gemm_acl", element::i8)}};

INSTANTIATE_TEST_SUITE_P(smoke_FCAndFQ_CPU,
                         FCAndFQ,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(filterCPUInfo(cpuParams)),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         FCAndFQ::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
