// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_transpose_reorder.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string FuseTransposeAndReorderTest::getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj) {
    std::ostringstream result;
    SizeVector inputShape;
    Precision inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "Precision=" << inPrec.name();

    return result.str();
}

void FuseTransposeAndReorderTest::CheckTransposeCount(size_t expectedTransposeCount) {
    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualTransposeCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Transpose") {
            actualTransposeCount++;
        }
    }

    ASSERT_EQ(expectedTransposeCount, actualTransposeCount);
}

void FuseTransposeAndReorderTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    std::tie(inputShape, inPrec) = this->GetParam();
    CreateGraph();
}

const auto fuseTransposeAndReorderCommonParams = ::testing::Combine(
        ::testing::Values(SizeVector{1, 2, 3, 4}, SizeVector{1, 2, 3, 4, 5}),
        ::testing::Values(Precision::I8, Precision::U8)
);

/*  FuseTransposeAndReorderTest graph
      ---------
      |Input  |
      ---------
          |
    -------------
    | --------- |
    | |Transpose| |
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

void FuseTransposeAndReorderTest::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};
    auto memFmt = inputShape.size() == 5 ? ndhwc : nhwc;

    auto constOrder = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(transpose)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(0);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FuseTransposeAndReorderTest, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);


/*  FuseTransposeAndReorderTest1 graph
             ---------
             |Input  |
             ---------
                 |
             ---------
             |Transpose|
             ---------
                 |
        -------------------
        |                 |
        |           -------------
        |           | --------- |
        |           | |Transpose| |
    ---------       | --------- |
    |Reshape|       |     |     |
    ---------       | --------- |
        |           | |Reorder| |
        |           | --------- |
        |           |-----------|
        |                 |
        |             ---------
        |             |Transpose|
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

void FuseTransposeAndReorderTest1::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};

    auto constOrder1 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose1 = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder1);
    auto memFmt1 = inputShape.size() == 5 ? ndhwc : nhwc;
    transpose1->get_rt_info() = makeCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose2 = std::make_shared<ngraph::opset5::Transpose>(transpose1, constOrder2);
    auto memFmt2 = inputShape.size() == 5 ? ndhwc : nhwc;
    transpose2->get_rt_info() = makeCPUInfo({memFmt2}, {memFmt2}, {});

    auto constOrder3 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose3 = std::make_shared<ngraph::opset5::Transpose>(transpose2, constOrder3);
    auto memFmt3 = inputShape.size() == 5 ? ncdhw : nchw;
    transpose3->get_rt_info() = makeCPUInfo({memFmt3}, {memFmt3}, {});

    auto shape = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, transpose3->get_output_shape(0));
    auto reshape = std::make_shared<ngraph::opset5::Reshape>(transpose1, shape, false);

    auto concat = ngraph::builder::makeConcat({transpose3, reshape}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Transpose_TransposeReorderTranspose_Reshape_Concat");
}

// Test disabled temporarily, it conflicts with TransposeFuse transformation in common optimizations step
TEST_P(FuseTransposeAndReorderTest1, DISABLED_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(2);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FuseTransposeAndReorderTest1, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);


/*  FuseTransposeAndReorderTest2 graph
    ---------         ---------
    |Input  |         |Input  |
    ---------         ---------
        |                 |
        |           -------------
    ---------       | ----------- |
    |Reorder|       | |Transpose| |
    ---------       | ----------- |
        |           |      |      |
    ---------       | ----------- |
    |Transpose|     |  |Reorder|  |
    ---------       | ----------- |
        |           |-------------|
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

void FuseTransposeAndReorderTest2::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);

    auto inputShape2(inputShape);
    inputShape2[inputShape2.size() - 1] *= 2;
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape2});

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 4, 1, 2, 3} : std::vector<int64_t>{0, 3, 1, 2};

    auto constOrder1 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose1 = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder1);
    auto memFmt1 = inputShape.size() == 5 ? ndhwc : nhwc;
    transpose1->get_rt_info() = makeCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose2 = std::make_shared<ngraph::opset5::Transpose>(params[1], constOrder2);
    auto memFmt2 = inputShape.size() == 5 ? ncdhw : nchw;
    transpose2->get_rt_info() = makeCPUInfo({memFmt2}, {memFmt2}, {});

    auto concat = ngraph::builder::makeConcat({transpose1, transpose2}, 1);
    concat->get_rt_info() = makeCPUInfo({memFmt1, memFmt1}, {memFmt1}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Transpose_Transpose_Concat");
}

TEST_P(FuseTransposeAndReorderTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckTransposeCount(1);
}

INSTANTIATE_TEST_CASE_P(smoke_Basic, FuseTransposeAndReorderTest2, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
