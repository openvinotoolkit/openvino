// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace CPULayerTestsDefinitions {

using tileCPUTestParams = std::tuple<SizeVector,           // Input shapes
                                     std::vector<int64_t>, // Repeats
                                     Precision>;           // Precision

class TileLayerCPUTest : public testing::WithParamInterface<tileCPUTestParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<tileCPUTestParams> obj) {
        SizeVector inputShapes;
        std::vector<int64_t> repeats;
        Precision precision;
        std::tie(inputShapes, repeats, precision) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "Repeats=" << CommonTestUtils::vec2str(repeats) << "_";
        result << "netPRC=" << precision.name();
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        SizeVector inputShapes;
        std::vector<int64_t> repeats;
        Precision precision;
        std::tie(inputShapes, repeats, precision) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto tile = ngraph::builder::makeTile(paramOuts[0], repeats);

        // WA: after removing tile node names for network outputs don't change
        auto mockNode = std::make_shared<ngraph::opset5::Multiply>(tile, ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1}));

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mockNode)};
        function = std::make_shared<ngraph::Function>(results, params, "tileCPU");
    }
};

TEST_P(TileLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

const std::vector<std::vector<int64_t>> repeats3D = {
        {1, 2, 3},
        {2, 1, 1},
        {2, 3, 1},
        {2, 2, 2},
        {1, 1, 1}
};

INSTANTIATE_TEST_CASE_P(smoke_3D, TileLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(SizeVector({2, 3, 4})),
                ::testing::ValuesIn(repeats3D),
                ::testing::Values(Precision::FP32)),
        TileLayerCPUTest::getTestCaseName);


const std::vector<std::vector<int64_t>> repeats4D = {
        {1, 2, 3, 4},
        {2, 1, 1, 4},
        {2, 3, 1, 4},
        {2, 2, 2, 4},
        {1, 1, 1, 1}
};

INSTANTIATE_TEST_CASE_P(smoke_4D, TileLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(SizeVector({2, 3, 4, 5})),
                ::testing::ValuesIn(repeats4D),
                ::testing::Values(Precision::FP32)),
        TileLayerCPUTest::getTestCaseName);

const std::vector<std::vector<int64_t>> repeats5D = {
        {1, 2, 3, 4, 4},
        {2, 1, 1, 4, 2},
        {2, 3, 1, 4, 5},
        {2, 2, 2, 4, 1},
        {1, 1, 1, 1, 1}
};

INSTANTIATE_TEST_CASE_P(smoke_5D, TileLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(SizeVector({1, 2, 3, 4, 6})),
                ::testing::ValuesIn(repeats5D),
                ::testing::Values(Precision::FP32)),
        TileLayerCPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
