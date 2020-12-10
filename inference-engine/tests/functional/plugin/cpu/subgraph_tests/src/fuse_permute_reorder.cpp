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

void FusePermuteAndReorderTest::CheckPermuteCount(size_t expectedPermuteCount) {
    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualPermuteCount = 0;
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
            actualPermuteCount++;
        }
    }

    ASSERT_EQ(expectedPermuteCount, actualPermuteCount);
}

void FusePermuteAndReorderTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    std::tie(inputShape, inPrec) = this->GetParam();
    CreateGraph();
}

const auto fusePermuteAndReorderCommonParams = ::testing::Combine(
        ::testing::Values(SizeVector{1, 2, 3, 4}, SizeVector{1, 2, 3, 4, 5}),
        ::testing::Values(Precision::I8, Precision::U8)
);

/*  FusePermuteAndReorderTest graph
      ---------
      |Input  |
      ---------
          |
    -------------
    | --------- |
    | |Permute| |
    | --------- |
    |     |     |
    | --------- |
    | |Reorder| |
    | --------- |
    |-----------|
          |
      ---------
      |Output |
      ---------
*/

void FusePermuteAndReorderTest::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};
    auto memFmt = inputShape.size() == 5 ? ndhwc : nhwc;

    auto constOrder = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder);
    permute->get_rt_info() = setCPUInfo({memFmt}, {memFmt}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(permute)};
    function = std::make_shared<ngraph::Function>(results, params, "PermuteReorder");
}

TEST_P(FusePermuteAndReorderTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPermuteCount(0);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FusePermuteAndReorderTest, fusePermuteAndReorderCommonParams, FusePermuteAndReorderTest::getTestCaseName);


/*  FusePermuteAndReorderTest1 graph
             ---------
             |Input  |
             ---------
                 |
             ---------
             |Permute|
             ---------
                 |
        -------------------
        |                 |
        |           -------------
        |           | --------- |
        |           | |Permute| |
    ---------       | --------- |
    |Reshape|       |     |     |
    ---------       | --------- |
        |           | |Reorder| |
        |           | --------- |
        |           |-----------|
        |                 |
        |             ---------
        |             |Permute|
        |             ---------
        |                 |
        --------   --------
               |   |
             ---------
             |Concat |
             ---------
                 |
             ---------
             |Output |
             ---------
*/

void FusePermuteAndReorderTest1::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};

    auto constOrder1 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute1 = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder1);
    auto memFmt1 = inputShape.size() == 5 ? ndhwc : nhwc;
    permute1->get_rt_info() = setCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute2 = std::make_shared<ngraph::opset5::Transpose>(permute1, constOrder2);
    auto memFmt2 = inputShape.size() == 5 ? ndhwc : nhwc;
    permute2->get_rt_info() = setCPUInfo({memFmt2}, {memFmt2}, {});

    auto constOrder3 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute3 = std::make_shared<ngraph::opset5::Transpose>(permute2, constOrder3);
    auto memFmt3 = inputShape.size() == 5 ? ncdhw : nchw;
    permute3->get_rt_info() = setCPUInfo({memFmt3}, {memFmt3}, {});

    auto shape = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, permute3->get_output_shape(0));
    auto reshape = std::make_shared<ngraph::opset5::Reshape>(permute1, shape, false);

    auto concat = ngraph::builder::makeConcat({permute3, reshape}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Permute_PermuteReorderPermute_Reshape_Concat");
}

TEST_P(FusePermuteAndReorderTest1, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPermuteCount(2);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FusePermuteAndReorderTest1, fusePermuteAndReorderCommonParams, FusePermuteAndReorderTest::getTestCaseName);


/*  FusePermuteAndReorderTest2 graph
    ---------         ---------
    |Input  |         |Input  |
    ---------         ---------
        |                 |
        |           -------------
    ---------       | --------- |
    |Reorder|       | |Permute| |
    ---------       | --------- |
        |           |     |     |
    ---------       | --------- |
    |Permute|       | |Reorder| |
    ---------       | --------- |
        |           |-----------|
        |                 |
        --------   --------
               |   |
             ---------
             |Concat |
             ---------
                 |
             ---------
             |Output |
             ---------
*/

void FusePermuteAndReorderTest2::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);

    auto inputShape2(inputShape);
    inputShape2[inputShape2.size() - 1] *= 2;
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape2});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 4, 1, 2, 3} : std::vector<int64_t>{0, 3, 1, 2};

    auto constOrder1 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute1 = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder1);
    auto memFmt1 = inputShape.size() == 5 ? ndhwc : nhwc;
    permute1->get_rt_info() = setCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto permute2 = std::make_shared<ngraph::opset5::Transpose>(params[1], constOrder2);
    auto memFmt2 = inputShape.size() == 5 ? ncdhw : nchw;
    permute2->get_rt_info() = setCPUInfo({memFmt2}, {memFmt2}, {});

    auto concat = ngraph::builder::makeConcat({permute1, permute2}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Permute_Permute_Concat");
}

TEST_P(FusePermuteAndReorderTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPermuteCount(1);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FusePermuteAndReorderTest2, fusePermuteAndReorderCommonParams, FusePermuteAndReorderTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
