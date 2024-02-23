// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/fuse_transpose_reorder.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"
#include "openvino/openvino.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string FuseTransposeAndReorderTest::getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj) {
    std::ostringstream result;
    ov::Shape input_shape;
    ov::element::Type in_prec;
    std::tie(input_shape, in_prec) = obj.param;

    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "Precision=" << in_prec.to_string();

    return result.str();
}

void FuseTransposeAndReorderTest::check_transpose_count(size_t expectedTransposeCount) {
    auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);
    size_t actual_transpose_count = 0;
    for (const auto &node : runtime_model->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        if (getExecValue(ov::exec_model_info::LAYER_TYPE) == "Transpose") {
            actual_transpose_count++;
        }
    }

    ASSERT_EQ(expectedTransposeCount, actual_transpose_count);
}

void FuseTransposeAndReorderTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(input_shape, in_prec) = this->GetParam();
    create_model();
}

const auto fuseTransposeAndReorderCommonParams = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 2, 3, 4}, ov::Shape{1, 2, 3, 4, 5}),
        ::testing::Values(ov::element::i8, ov::element::u8)
);

/*  FuseTransposeAndReorderTest model
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

void FuseTransposeAndReorderTest::create_model() {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};

    auto order = input_shape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};
    auto memFmt = input_shape.size() == 5 ? ndhwc : nhwc;

    auto constOrder = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};
    function = std::make_shared<ov::Model>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest, CompareWithRefs) {
    run();
    check_transpose_count(0);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest, fuseTransposeAndReorderCommonParams, FuseTransposeAndReorderTest::getTestCaseName);


/*  FuseTransposeAndReorderTest1 model
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

void FuseTransposeAndReorderTest1::create_model() {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};
    auto order = input_shape.size() == 5 ? std::vector<int64_t>{0, 2, 3, 4, 1} : std::vector<int64_t>{0, 2, 3, 1};

    auto constOrder1 = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder1);
    auto memFmt1 = input_shape.size() == 5 ? ndhwc : nhwc;
    transpose1->get_rt_info() = makeCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(transpose1, constOrder2);
    auto memFmt2 = input_shape.size() == 5 ? ndhwc : nhwc;
    transpose2->get_rt_info() = makeCPUInfo({memFmt2}, {memFmt2}, {});

    auto constOrder3 = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(transpose2, constOrder3);
    auto memFmt3 = input_shape.size() == 5 ? ncdhw : nchw;
    transpose3->get_rt_info() = makeCPUInfo({memFmt3}, {memFmt3}, {});

    auto shape = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, transpose3->get_output_shape(0));
    auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose1, shape, false);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{transpose3, reshape}, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, params, "Transpose_TransposeReorderTranspose_Reshape_Concat");
}

// Test disabled temporarily, it conflicts with TransposeFuse transformation in common optimizations step
TEST_P(FuseTransposeAndReorderTest1, DISABLED_CompareWithRefs) {
    run();
    check_transpose_count(2);
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

void FuseTransposeAndReorderTest2::create_model() {
    auto input_shape2(input_shape);
    input_shape2[input_shape2.size() - 1] *= 2;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape)),
                               std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape2))};
    auto order = input_shape.size() == 5 ? std::vector<int64_t>{0, 4, 1, 2, 3} : std::vector<int64_t>{0, 3, 1, 2};

    auto constOrder1 = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(params[0], constOrder1);
    auto memFmt1 = input_shape.size() == 5 ? ndhwc : nhwc;
    transpose1->get_rt_info() = makeCPUInfo({memFmt1}, {memFmt1}, {});

    auto constOrder2 = ov::test::utils::deprecated::make_constant(ov::element::i64, {input_shape.size()}, order);
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(params[1], constOrder2);
    auto memFmt2 = input_shape.size() == 5 ? ncdhw : nchw;
    transpose2->get_rt_info() = makeCPUInfo({memFmt2}, {memFmt2}, {});

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{transpose1, transpose2}, 1);
    concat->get_rt_info() = makeCPUInfo({memFmt1, memFmt1}, {memFmt1}, {});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, params, "Transpose_Transpose_Concat");
}

TEST_P(FuseTransposeAndReorderTest2, CompareWithRefs) {
    run();
    check_transpose_count(1);
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

void FuseTransposeAndReorderTest3::create_model() {
    OPENVINO_ASSERT(input_shape.size() == 4);

    auto memFmt = nhwc;
    ov::op::PadType padType = ov::op::PadType::SAME_UPPER;
    ov::Shape kernel{3, 3}, stride{1, 1}, dilation{1, 1};
    std::vector<ptrdiff_t> padBegin{0, 0}, padEnd{0, 0};
    size_t convOutChannels = 32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};
    OPENVINO_ASSERT(input_shape[1] >= 8 && (input_shape[1] % 8 == 0));
    auto convolutionNode = ov::test::utils::make_convolution(params.front(),
                                                             in_prec,
                                                             kernel,
                                                             stride,
                                                             padBegin,
                                                             padEnd,
                                                             dilation,
                                                             padType,
                                                             convOutChannels);
    convolutionNode->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    auto sndAddIn = std::make_shared<ov::op::v0::Parameter>(in_prec, convolutionNode->get_output_shape(0));
    params.push_back(sndAddIn);
    auto add = std::make_shared<ov::op::v1::Add>(convolutionNode->output(0), sndAddIn);

    auto order = std::vector<int64_t>{0, 2, 3, 1};
    auto constOrder = ov::test::utils::deprecated::make_constant(ov::element::i64, {order.size()}, order);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(add, constOrder);
    transpose->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};
    function = std::make_shared<ov::Model>(results, params, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest3, CompareWithRefs) {
    run();
    check_transpose_count(1);
}

const auto convSumTranposeParams = ::testing::Combine(::testing::Values(ov::Shape{1, 16, 32, 35}),
                                                      ::testing::Values(ov::element::f32)
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
void FuseTransposeAndReorderTest4::create_model() {
    OPENVINO_ASSERT(input_shape.size() == 4);
    const ov::Shape kernel = {1, 1};
    const ov::Shape stride = {1, 1};
    const ov::Shape dilation = {1, 1};
    const std::vector<ptrdiff_t> padBegin = {0, 0};
    const std::vector<ptrdiff_t> padEnd = {0, 0};
    const size_t convOutChannels = 4;
    auto memFmt = nhwc;

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};
    const auto relu = std::make_shared<ov::op::v0::Relu>(inputParams[0]);
    const auto transposeOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
    const auto transpose1 = std::make_shared<ov::op::v1::Transpose>(relu, transposeOrder);
    const auto conv1 = ov::test::utils::make_convolution(transpose1,
                                                         in_prec,
                                                         kernel,
                                                         stride,
                                                         padBegin,
                                                         padEnd,
                                                         dilation,
                                                         ov::op::PadType::AUTO,
                                                         convOutChannels);
    conv1->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(relu, transposeOrder);
    const auto conv2 = ov::test::utils::make_convolution(transpose2,
                                                         in_prec,
                                                         kernel,
                                                         stride,
                                                         padBegin,
                                                         padEnd,
                                                         dilation,
                                                         ov::op::PadType::AUTO,
                                                         convOutChannels);
    conv2->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto add = std::make_shared<ov::op::v1::Add>(conv1, conv2);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add->output(0))};
    function = std::make_shared<ov::Model>(results, inputParams, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest4, CompareWithRefs) {
    run();
    check_transpose_count(0);
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest4, convSumTranposeParams, FuseTransposeAndReorderTest::getTestCaseName);

void FuseTransposeAndReorderTest5::create_model() {
    OPENVINO_ASSERT(input_shape.size() == 4);
    const ov::Shape kernel = {1, 1};
    const ov::Shape stride = {1, 1};
    const ov::Shape dilation = {1, 1};
    const std::vector<ptrdiff_t> padBegin = {0, 0};
    const std::vector<ptrdiff_t> padEnd = {0, 0};
    const size_t convOutChannels = 4;
    auto memFmt = nhwc;

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(input_shape))};
    const auto relu = std::make_shared<ov::op::v0::Relu>(inputParams[0]);
    const auto transposeOrder = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
    const auto transpose_shared = std::make_shared<ov::op::v1::Transpose>(relu, transposeOrder);
    const auto conv1 = ov::test::utils::make_convolution(transpose_shared,
                                                         in_prec,
                                                         kernel,
                                                         stride,
                                                         padBegin,
                                                         padEnd,
                                                         dilation,
                                                         ov::op::PadType::AUTO,
                                                         convOutChannels);
    conv1->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto conv2 = ov::test::utils::make_convolution(transpose_shared,
                                                         in_prec,
                                                         kernel,
                                                         stride,
                                                         padBegin,
                                                         padEnd,
                                                         dilation,
                                                         ov::op::PadType::AUTO,
                                                         convOutChannels);
    conv2->get_rt_info() = makeCPUInfo({memFmt}, {memFmt}, {});
    const auto add = std::make_shared<ov::op::v1::Add>(conv1, conv2);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add->output(0))};
    function = std::make_shared<ov::Model>(results, inputParams, "TransposeReorder");
}

TEST_P(FuseTransposeAndReorderTest5, CompareWithRefs) {
    run();
    check_transpose_count(0);
}
INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseTransposeAndReorderTest5, convSumTranposeParams, FuseTransposeAndReorderTest::getTestCaseName);

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

}  // namespace test
}  // namespace ov
