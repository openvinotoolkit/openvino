// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_transpose_reorder.hpp"
#include <ov_models/preprocess/preprocess_builders.hpp>
#include <openvino/openvino.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string FuseTransposeAndReorderTest::getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj) {
    std::ostringstream result;
    SizeVector inputShape;
    Precision inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
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
            return it->second.as<std::string>();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Transpose") {
            actualTransposeCount++;
        }
    }

    ASSERT_EQ(expectedTransposeCount, actualTransposeCount);
}

void FuseTransposeAndReorderTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};
    auto memFmt = inputShape.size() == 5 ? ndhwc : nhwc;

    auto constOrder = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(transpose)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest, CompareWithRefs) {
    Run();
    CheckTransposeCount(0);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);


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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

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

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{transpose3, reshape}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Transpose_TransposeReorderTranspose_Reshape_Concat");
}

// Test disabled temporarily, it conflicts with TransposeFuse transformation in common optimizations step
TEST_P(FuseTransposeAndReorderTest1, DISABLED_CompareWithRefs) {
    Run();
    CheckTransposeCount(2);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest1, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);


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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape2))};
    auto order = inputShape.size() == 5 ? std::vector<int64_t>{0, 4, 1, 2, 3} : std::vector<int64_t>{0, 3, 1, 2};

    auto constOrder1 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose1 = std::make_shared<ngraph::opset5::Transpose>(params[0], constOrder1);
    auto memFmt1 = inputShape.size() == 5 ? ndhwc : nhwc;
    transpose1->get_rt_info() = makeCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ngraph::builder::makeConstant(ngraph::element::i64, {inputShape.size()}, order);
    auto transpose2 = std::make_shared<ngraph::opset5::Transpose>(params[1], constOrder2);
    auto memFmt2 = inputShape.size() == 5 ? ncdhw : nchw;
    transpose2->get_rt_info() = makeCPUInfo({memFmt2}, {memFmt2}, {});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{transpose1, transpose2}, 1);
    concat->get_rt_info() = makeCPUInfo({memFmt1, memFmt1}, {memFmt1}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "Transpose_Transpose_Concat");
}

TEST_P(FuseTransposeAndReorderTest2, CompareWithRefs) {
    Run();
    CheckTransposeCount(1);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest2, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);

/*  FuseTransposeAndReorderTest3 graph
    Parameter
        \
         \
       Convolution (nhwc)
           \
            \  Parameter
             \ /
             Add
              |
          Transpose (0,2,3,1)
              |
            Result
*/

void FuseTransposeAndReorderTest3::CreateGraph() {
    IE_ASSERT(inputShape.size() == 4);

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);

    auto memFmt = nhwc;
    ngraph::op::PadType padType = ngraph::op::PadType::SAME_UPPER;
    InferenceEngine::SizeVector kernel{3, 3}, stride{1, 1}, dilation{1, 1};
    std::vector<ptrdiff_t> padBegin{0, 0}, padEnd{0, 0};
    size_t convOutChannels = 32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    IE_ASSERT(inputShape[1] >= 8 && (inputShape[1] % 8 == 0));

    auto convolutionNode = ngraph::builder::makeConvolution(params.front(), ngPrc, kernel, stride, padBegin,
                                                            padEnd, dilation, padType, convOutChannels);
    convolutionNode->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    auto sndAddIn = std::make_shared<ngraph::opset1::Parameter>(ngPrc, convolutionNode->get_output_shape(0));
    params.push_back(sndAddIn);
    auto add = std::make_shared<ngraph::opset1::Add>(convolutionNode->output(0), sndAddIn);

    auto order = std::vector<int64_t>{0, 2, 3, 1};
    auto constOrder = ngraph::builder::makeConstant(ngraph::element::i64, {order.size()}, order);
    auto transpose = std::make_shared<ngraph::opset5::Transpose>(add, constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(transpose)};
    function = std::make_shared<ngraph::Function>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest3, CompareWithRefs) {
    Run();
    CheckTransposeCount(1);
}

const auto convSumTranposeParams = ::testing::Combine(::testing::Values(SizeVector{1, 16, 32, 35}),
                                                      ::testing::Values(Precision::FP32)
);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest3, convSumTranposeParams, FuseTransposeAndReorderTest::getTestCaseName);

/*  FuseTransposeAndReorderTest4 graph
         param
           |
          relu
           |
     ---------------
     |  const      |
     |   |---- transpose
    transpose      |
         |    convolution
         |         |
    convolution    |
         |         |
         |--------add
                   |
                 result
*/
void FuseTransposeAndReorderTest4::CreateGraph() {
    IE_ASSERT(inputShape.size() == 4);
    const InferenceEngine::SizeVector kernel = {1, 1};
    const InferenceEngine::SizeVector stride = {1, 1};
    const InferenceEngine::SizeVector dilation = {1, 1};
    const std::vector<ptrdiff_t> padBegin = {0, 0};
    const std::vector<ptrdiff_t> padEnd = {0, 0};
    const size_t convOutChannels = 4;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto memFmt = nhwc;

    ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    const auto relu = std::make_shared<ov::op::v0::Relu>(inputParams[0]);
    const auto transposeOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(relu, transposeOrder);
    const auto conv1 = ngraph::builder::makeConvolution(transpose1, ngPrc, kernel, stride, padBegin,
                                                    padEnd, dilation, ngraph::op::PadType::AUTO, convOutChannels);
    conv1->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(relu, transposeOrder);
    const auto conv2 = ngraph::builder::makeConvolution(transpose2, ngPrc, kernel, stride, padBegin,
                                                    padEnd, dilation, ngraph::op::PadType::AUTO, convOutChannels);
    conv2->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto add = std::make_shared<ov::op::v1::Add>(conv1, conv2);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add->output(0))};
    function = std::make_shared<ov::Model>(results, inputParams, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest4, CompareWithRefs) {
    Run();
    CheckTransposeCount(0);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest4, convSumTranposeParams, FuseTransposeAndReorderTest::getTestCaseName);

TEST(smoke_Basic, FuseDynamicTransposeAndReorderTest) {
    auto model = ov::builder::preprocess::create_preprocess_1input(ov::element::u8, ov::PartialShape{1, 3, 224, 224});
    auto p = ov::preprocess::PrePostProcessor(model);
    p.input().tensor().set_spatial_dynamic_shape().set_layout("NHWC");
    p.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    model = p.build();

    auto core = ov::Core();
    ASSERT_NO_THROW(core.compile_model(model, "CPU"));
}

}  // namespace SubgraphTestsDefinitions
