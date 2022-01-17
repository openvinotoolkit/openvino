// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

struct TopKTestCase {
    DataShapeWithUpperBound dataShapes;
    int64_t k;
    int64_t axis;
};

const auto combinations = testing::Combine(
    testing::Values(
            ngraph::element::f16),
    testing::Values(
            ngraph::element::i32),
    testing::Values(
            TopKTestCase{{{12345}, {80000}}, 75, 0},
            TopKTestCase{{{1234}, {4663}}, 70, 0},
            TopKTestCase{{{1234}, {4663}}, 70, -1}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    DataType,
    DataType,
    TopKTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_TopK_Const : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& topkSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, topkSetup.dataShapes);
        const auto k = ngraph::opset3::Constant::create(idxType, {}, std::vector<int64_t>{topkSetup.k});

        return std::make_shared<ngraph::opset3::TopK>(inputSubgraph, k, topkSetup.axis, "max", "value");
    }
};

TEST_P(DSR_TopK_Const, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTopKConst, DSR_TopK_Const, combinations);

class DSR_TopK : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& topkSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, topkSetup.dataShapes);

        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(inputSubgraph->input_value(0), ngraph::element::i32);
        const auto gather = std::make_shared<ngraph::opset3::Gather>(
                shapeOf,
                ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {topkSetup.axis}),
                ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}));
        const auto upper_bound = ngraph::opset3::Constant::create(inputSubgraph->get_input_element_type(1), {1}, {topkSetup.k});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{upper_bound, gather}, 0);
        const auto k = std::make_shared<ngraph::opset3::ReduceMin>(
                concat, ngraph::opset3::Constant::create(ngraph::element::i32, {1}, {0}), false);

        return std::make_shared<ngraph::opset3::TopK>(inputSubgraph, k, topkSetup.axis, "max", "value");
    }
};

TEST_P(DSR_TopK, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicTopKConst, DSR_TopK, combinations);

}  // namespace
