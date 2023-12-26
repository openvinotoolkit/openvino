// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>

#include <ie_core.hpp>
#include <ngraph/function.hpp>
#include <exec_graph_info.hpp>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "execution_graph_tests/fp16_compress_disable.hpp"
#include "ie/ie_plugin_config.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphDisableFP16Compress::getTestCaseName(testing::TestParamInfo<ExecGraphDisableFP16CompressSpecificParams> obj) {
    std::ostringstream result;
    bool matmulFP16Disabled;
    bool FCFP16Disabled;
    std::string targetDevice;
    std::tie(matmulFP16Disabled, FCFP16Disabled, targetDevice) = obj.param;
    result << "matmul_BF16_Disabled=" << matmulFP16Disabled << "_";
    result << "FC_BF16_Disabled=" << FCFP16Disabled << "_";
    result << "device=" << targetDevice;
    return result.str();
}

void ExecGraphDisableFP16Compress::SetUp() {
    std::tie(matmulFP16Disabled, FCFP16Disabled, targetDevice) =  this->GetParam();
    create_model();
}

void ExecGraphDisableFP16Compress::TearDown() {
}

/*  test model:
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
    | |  FC   |
    | --------- |
    |-----------|
          |
      ---------
      |Output |
      ---------
*/

void ExecGraphDisableFP16Compress::create_model() {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 16});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16, 16});
    ov::ParameterVector inputParams = {A , B};
    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);
    matmul->set_friendly_name("Matmul0");
    if (matmulFP16Disabled)
        ov::disable_fp16_compression(matmul);
    auto fcWeight = ngraph::builder::makeConstant(ov::element::f16, ov::Shape{16, 16}, std::vector<float>{0.0f}, true);
    auto weightConvert = std::make_shared<ov::op::v0::Convert>(fcWeight, ov::element::f32);
    auto fc = std::make_shared<ov::op::v0::MatMul>(matmul, weightConvert);
    fc->set_friendly_name("FC0");
    if (FCFP16Disabled)
        ov::disable_fp16_compression(fc);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fc)};
    function = std::make_shared<ov::Model>(results, inputParams, "testModel");
}

void ExecGraphDisableFP16Compress::checkInferPrecision() {
    auto ie  = InferenceEngine::Core();
    auto net = InferenceEngine::CNNNetwork(function);
    InferenceEngine::ExecutableNetwork exec_net;
    if (targetDevice == "CPU") {
        //Enforce CPU to infer with BF16 precision.
        exec_net = ie.LoadNetwork(net, targetDevice,
                                    {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}});
        enforcedPrecision = "bf16";
    } else {
        exec_net = ie.LoadNetwork(net, targetDevice);
        enforcedPrecision = "f16";
    }

    auto runtime_model = exec_net.GetExecGraphInfo().getFunction();

    ASSERT_NE(nullptr, runtime_model);

    auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
        auto it = rtInfo.find(paramName);
        OPENVINO_ASSERT(rtInfo.end() != it);
        return it->second.as<std::string>();
    };
    std::string matmulPrecision, fcPrecision;
    for (const auto &node : runtime_model->get_ops()) {
        if (getExecValue(node->get_rt_info(), ov::exec_model_info::ORIGINAL_NAMES) == "Matmul0") {
            matmulPrecision = getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION);
        } else if (getExecValue(node->get_rt_info(), ov::exec_model_info::ORIGINAL_NAMES) == "FC0") {
            fcPrecision = getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION);
        }
    }
    OPENVINO_ASSERT(!matmulPrecision.empty() && !fcPrecision.empty());
    if (matmulFP16Disabled)
        ASSERT_EQ(matmulPrecision, "f32");
    else
        ASSERT_EQ(matmulPrecision, enforcedPrecision);

    if (FCFP16Disabled)
        ASSERT_EQ(fcPrecision, "f32");
    else
        ASSERT_EQ(fcPrecision, enforcedPrecision);
}

TEST_P(ExecGraphDisableFP16Compress, CheckRuntimePrecision) {
    // Only run tests on CPU with avx512_core ISA
    if (targetDevice == "CPU" && !ov::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    checkInferPrecision();
}

}  // namespace ExecutionGraphTests
