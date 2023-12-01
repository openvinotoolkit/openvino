// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <vector>
#include <string>
#include "test_utils/cpu_test_utils.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/matmul.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"


using namespace CPUTestUtils;

namespace ov {
namespace test {
using InferPrecisionBF16DisableTestParams = std::tuple<bool, // Matmul BF16 disabled.
                                                       bool  // FC BF16 disabled.
                                        >;

class InferPrecisionBF16DisableTest : public testing::WithParamInterface<InferPrecisionBF16DisableTestParams>, public CPUTestsBase,
        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferPrecisionBF16DisableTestParams> obj);

protected:
    void SetUp() override;
    virtual void create_model();
    void checkInferPrecision();

    bool matmulBF16Disabled;
    bool FCBF16Disabled;
};


std::string InferPrecisionBF16DisableTest::getTestCaseName(testing::TestParamInfo<InferPrecisionBF16DisableTestParams> obj) {
    std::ostringstream result;
    bool matmulBF16Disabled;
    bool FCBF16Disabled;
    std::tie(matmulBF16Disabled, FCBF16Disabled) = obj.param;
    result << "matmul_BF16_Disabled=" << matmulBF16Disabled << "_";
    result << "FC_BF16_Disabled=" << FCBF16Disabled;
    return result.str();
}

void InferPrecisionBF16DisableTest::checkInferPrecision() {
    auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);

    auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
        auto it = rtInfo.find(paramName);
        OPENVINO_ASSERT(rtInfo.end() != it);
        return it->second.as<std::string>();
    };

    for (const auto &node : runtime_model->get_ops()) {
        if (getExecValue(node->get_rt_info(), ov::exec_model_info::LAYER_TYPE) == "MatMul") {
            if (matmulBF16Disabled)
                ASSERT_EQ(getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION), "f32");
            else
                ASSERT_EQ(getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION), "bf16");
        } else if (getExecValue(node->get_rt_info(), ov::exec_model_info::LAYER_TYPE) == "FullyConnected") {
            if (FCBF16Disabled)
                ASSERT_EQ(getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION), "f32");
            else
                ASSERT_EQ(getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION), "bf16");
        }
    }
}

void InferPrecisionBF16DisableTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(matmulBF16Disabled, FCBF16Disabled) = this->GetParam();
    create_model();
    //Enforce BF16 in this test.
    configuration.insert({ InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES });
    rel_threshold = 1e-2;
}

const auto InferPrecisionBF16DisableTestCommonParams = ::testing::Combine(
        ::testing::Values(true, false),
        ::testing::Values(true, false)
);

/*  InferPrecisionBF16DisableTest model
      ---------
      |Input  |
      ---------
          |
    -------------
    | --------- |
    | | Matmul| |
    | --------- |
    |     |     |
    | --------- |
    | |FC+Bias| |
    | --------- |
    |-----------|
          |
      ---------
      |Output |
      ---------
*/

void InferPrecisionBF16DisableTest::create_model() {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 16});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16, 16});
    ov::ParameterVector inputParams = {A , B};
    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);
    matmul->set_friendly_name("Matmul0");
    if (matmulBF16Disabled)
        ov::disable_fp16_compression(matmul);
    auto fcWeight = ngraph::builder::makeConstant(ov::element::f16, ov::Shape{16, 16}, std::vector<float>{0.0f}, true);
    auto weightConvert = std::make_shared<ngraph::opset1::Convert>(fcWeight, ov::element::f32);
    auto fc = std::make_shared<ngraph::opset1::MatMul>(matmul, weightConvert);
    fc->set_friendly_name("FC0");
    if (FCBF16Disabled)
        ov::disable_fp16_compression(fc);
    auto bias = ngraph::builder::makeConstant<float>(ngraph::element::Type_t::f16, ngraph::Shape({16}), {}, true);
    auto biasConvert = std::make_shared<ngraph::opset1::Convert>(bias, ov::element::f32);
    auto biasAdd = std::make_shared<ngraph::opset3::Add>(fc, biasConvert);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(biasAdd)};
    function = std::make_shared<ov::Model>(results, inputParams, "SubgraphDisableBF16");
}

TEST_P(InferPrecisionBF16DisableTest, CompareWithRefs) {
    //The test aims to check disabling BF16 , so will not run for the platform without BF16 support.
    if (!InferenceEngine::with_cpu_x86_bfloat16() && !InferenceEngine::with_cpu_x86_avx512_core_amx_bf16())
        GTEST_SKIP();
    run();
    checkInferPrecision();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, InferPrecisionBF16DisableTest,
                        InferPrecisionBF16DisableTestCommonParams, InferPrecisionBF16DisableTest::getTestCaseName);
}  // namespace test
}  // namespace ov
