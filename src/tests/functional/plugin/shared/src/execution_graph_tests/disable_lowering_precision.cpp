// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>

#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/core/model.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "execution_graph_tests/disable_lowering_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphDisableLoweringPrecision::getTestCaseName(testing::TestParamInfo<ExecGraphDisableLoweringPrecisionSpecificParams> obj) {
    std::ostringstream result;
    bool disableLoweringPrecision;
    std::string targetDevice;
    ov::element::Type loweringPrecision;

    std::tie(disableLoweringPrecision, targetDevice, loweringPrecision) = obj.param;
    result << "matmul_disable_lowingprecision=" << disableLoweringPrecision << "_";
    result << "device=" << targetDevice << "_";
    result << "loweringPrecision=" << loweringPrecision.to_string();
    return result.str();
}

void ExecGraphDisableLoweringPrecision::SetUp() {
    std::tie(disableLoweringPrecision, targetDevice, loweringPrecision) =  this->GetParam();
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
    auto core = ov::test::utils::PluginCache::get().core();
    compiledModel = core->compile_model(funcPtr, targetDevice,
                                    ov::hint::inference_precision(loweringPrecision));
    const auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);
    auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
        auto it = rtInfo.find(paramName);
        OPENVINO_ASSERT(rtInfo.end() != it);
        return it->second.as<std::string>();
    };
    std::string matmulPrecision;
    for (const auto &node : runtime_model->get_ops()) {
        const auto origName = getExecValue(node->get_rt_info(), ov::exec_model_info::ORIGINAL_NAMES);
        if (origName.find("Matmul0") != std::string::npos) {
            matmulPrecision = getExecValue(node->get_rt_info(), ov::exec_model_info::RUNTIME_PRECISION);
            break;
        }
    }
    ASSERT_TRUE(!matmulPrecision.empty());
    if (disableLoweringPrecision)
        ASSERT_EQ(matmulPrecision, "f32");
    else
        ASSERT_EQ(matmulPrecision, loweringPrecision.to_string());
    funcPtr.reset();
}

TEST_P(ExecGraphDisableLoweringPrecision, CheckRuntimePrecision) {
    checkInferPrecision();
}
}  // namespace ExecutionGraphTests
