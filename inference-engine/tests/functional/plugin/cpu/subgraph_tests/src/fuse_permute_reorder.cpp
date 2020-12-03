// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_permute_reorder.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

std::string FusePermuteAndReorderTest::getTestCaseName(testing::TestParamInfo<FusePermuteAndReorderParams> obj) {
    std::ostringstream result;
    SizeVector inputShape;
    Precision inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "Precision=" << inPrec.name();

    return result.str();
}

void FusePermuteAndReorderTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;
    SizeVector inputShape;
    Precision inPrec;

    std::tie(inputShape, inPrec) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};
    auto memFmt = inputShape.size() == 5 ? ndhwc : nhwc;

    auto constOrder = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);

    auto permute = std::make_shared<ngraph::opset5::Transpose>(paramOuts[0], constOrder);

    permute->get_rt_info() = setCPUInfo({memFmt}, {memFmt}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(permute)};
    function = std::make_shared<ngraph::Function>(results, params, "PermuteReorder");
}

TEST_P(FusePermuteAndReorderTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();

    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    bool permuteFound = false;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Permute") {
            permuteFound = true;
            break;
        }
    }
    ASSERT_TRUE(!permuteFound);
}

const auto fusePermuteAndReorderParams = ::testing::Combine(
        ::testing::Values(SizeVector{1, 2, 3, 4}, SizeVector{1, 2, 3, 4, 5}),
        ::testing::Values(Precision::I8, Precision::U8)
);

INSTANTIATE_TEST_CASE_P(smoke_Basic, FusePermuteAndReorderTest, fusePermuteAndReorderParams, FusePermuteAndReorderTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
