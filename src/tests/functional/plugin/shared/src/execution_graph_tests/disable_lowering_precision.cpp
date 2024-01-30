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
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/core/model.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "execution_graph_tests/disable_lowering_precision.hpp"
#include "ie/ie_plugin_config.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphDisableLoweringPrecision::getTestCaseName(testing::TestParamInfo<ExecGraphDisableLowingPrecisionSpecificParams> obj) {
    std::ostringstream result;
    bool disableLoweringPrecision;
    std::string targetDevice;
    std::tie(disableLoweringPrecision, targetDevice) = obj.param;
    result << "matmul_disable_lowingprecision=" << disableLoweringPrecision << "_";
    result << "device=" << targetDevice;
    return result.str();
}

void ExecGraphDisableLoweringPrecision::SetUp() {
    std::tie(disableLoweringPrecision, targetDevice) =  this->GetParam();
    create_model();
}

void ExecGraphDisableLoweringPrecision::TearDown() {
}

/*  test model:
      ---------
      |Input  |
      ---------
          |
      ---------
      |matmul |
      ---------
          |
      ---------
      |Output |
      ---------
*/

void ExecGraphDisableLoweringPrecision::create_model() {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 16});
    auto weiShape = ov::Shape{16, 16};
    auto weightConst = ov::op::v0::Constant::create(ov::element::i64, weiShape, std::vector<int64_t>(ov::shape_size(weiShape), 1));
    auto weightConvert = std::make_shared<ov::op::v0::Convert>(weightConst, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, weightConvert);
    matmul->set_friendly_name("Matmul0");
    if (disableLoweringPrecision)
        ov::disable_fp16_compression(matmul);
    funcPtr = std::make_shared<ov::Model>(matmul->outputs(), ov::ParameterVector{A}, "testModel");
}

void ExecGraphDisableLoweringPrecision::checkInferPrecision() {
    ov::CompiledModel compiledModel;
    std::string loweringPrecision;
    auto core = ov::test::utils::PluginCache::get().core();
    if (targetDevice == "CPU") {
        compiledModel = core->compile_model(funcPtr, targetDevice,
                                ov::hint::inference_precision(ov::element::bf16));
        loweringPrecision = "bf16";
    } else {
        compiledModel = core->compile_model(funcPtr, targetDevice);
        loweringPrecision = "f16";
    }
    const auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);
    auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
        auto it = rtInfo.find(paramName);
        OPENVINO_ASSERT(rtInfo.end() != it);
        return it->second.as<std::string>();
    };
    std::string matmulPrecision;
    for (const auto &node : runtime_model->get_ops()) {
        if (getExecValue(node->get_rt_info(), ov::exec_model_info::ORIGINAL_NAMES) == "Matmul0") {
            matmulPrecision = getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION);
        }
    }
    ASSERT_TRUE(!matmulPrecision.empty());
    if (disableLoweringPrecision)
        ASSERT_EQ(matmulPrecision, "f32");
    else
        ASSERT_EQ(matmulPrecision, loweringPrecision);
}

TEST_P(ExecGraphDisableLoweringPrecision, CheckRuntimePrecision) {
    // Only run tests on CPU with avx512_core ISA
    if (targetDevice == "CPU" && !ov::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    checkInferPrecision();
}
}  // namespace ExecutionGraphTests
