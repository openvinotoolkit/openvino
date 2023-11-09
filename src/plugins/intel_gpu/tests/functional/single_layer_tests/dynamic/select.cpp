// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
    std::vector<InputShape>,    // input shapes
    ElementType,                // presion of 'then' and 'else' of inputs
    op::AutoBroadcastSpec,      // broadcast spec
    TargetDevice                // device name
> SelectLayerTestParamSet;

class SelectLayerGPUTest : public testing::WithParamInterface<SelectLayerTestParamSet>,
                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectLayerTestParamSet>& obj) {
        std::vector<InputShape> inshapes;
        ElementType netType;
        op::AutoBroadcastSpec broadcast;
        TargetDevice targetDevice;
        std::tie(inshapes, netType, broadcast, targetDevice) = obj.param;

        std::ostringstream result;

        result << "IS=";
        for (const auto& shape : inshapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inshapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "Precision=" << netType << "_";
        result << "Broadcast=" << broadcast.m_type << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inshapes;
        ElementType netType;
        op::AutoBroadcastSpec broadcast;
        std::tie(inshapes, netType, broadcast, targetDevice) = this->GetParam();

        init_input_shapes(inshapes);

        ParameterVector params = {
            std::make_shared<opset1::Parameter>(ElementType::boolean, inputDynamicShapes[0]),
            std::make_shared<opset1::Parameter>(netType, inputDynamicShapes[1]),
            std::make_shared<opset1::Parameter>(netType, inputDynamicShapes[2]),
        };

        auto select = std::make_shared<ov::op::v1::Select>(params[0], params[1], params[2], broadcast);

        auto makeFunction = [](ParameterVector &params, const std::shared_ptr<Node> &lastNode) {
            ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<opset1::Result>(lastNode->output(i)));

            return std::make_shared<Function>(results, params, "SelectLayerGPUTest");
        };
        function = makeFunction(params, select);
    }
};

TEST_P(SelectLayerGPUTest, CompareWithRefs) {
   SKIP_IF_CURRENT_TEST_IS_DISABLED()

   run();
}

namespace {

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::f16,
        ElementType::i32,
};

namespace Select {

// AutoBroadcastType: NUMPY
const std::vector<std::vector<InputShape>> inShapesDynamicNumpy = {
    {
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} },
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} },
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} }
    },
    {
        { {-1, -1, -1, -1}, {{5, 1, 2, 1}, {1, 1, 9, 1}} },
        { {-1, -1, -1, -1}, {{1, 1, 1, 1}, {1, 1, 1, 1}} },
        { {-1, -1, -1, -1}, {{5, 1, 2, 1}, {9, 5, 9, 8}} }
    },
    {
        { {        -1, -1}, {{      8, 1}, {      8, 3}} },
        { {-1, -1, -1, -1}, {{2, 1, 8, 1}, {2, 9, 8, 1}} },
        { {    -1, -1, -1}, {{   9, 1, 1}, {   9, 1, 3}} }
    },
    {
        { {            -1}, {{         6}, {         3}} },
        { {-1, -1, -1, -1}, {{2, 1, 8, 1}, {2, 9, 8, 1}} },
        { {    -1, -1, -1}, {{   9, 1, 1}, {   9, 1, 3}} }
    },
    {
        { {-1, -1, -1, -1}, {{4, 1, 1, 1}, {5, 5, 5, 1}, {3, 4, 5, 6}} },
        { {-1, -1, -1, -1}, {{1, 8, 1, 1}, {1, 1, 1, 8}, {3, 4, 5, 6}} },
        { {    -1, -1, -1}, {{   1, 5, 1}, {   5, 5, 8}, {   4, 5, 6}} }
    },
    {
        { {-1, -1, -1, -1}, {{3, 4, 5, 6}} },
        { {    -1, -1, -1}, {{   4, 5, 6}} },
        { {        -1, -1}, {{      5, 6}} }
    },
};

const auto numpyCases = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamicNumpy),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(op::AutoBroadcastType::NUMPY),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_select_CompareWithRefsNumpy_dynamic, SelectLayerGPUTest, numpyCases, SelectLayerGPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inShapesDynamicRangeNumpy = {
    {
        { {                         {1, 10}}, {{         4}, {          10}, {         1}} },
        { {{2, 7}, {1, 6}, {5, 12}, {1, 20}}, {{5, 6, 6, 1}, {7, 6, 10, 10}, {5, 4, 5, 3}} },
        { {                {2, 10}, {1, 16}}, {{      6, 4}, {      10, 10}, {      5, 1}} }
    },
};

const auto rangeNumpyCases = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamicRangeNumpy),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(op::AutoBroadcastType::NUMPY),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_select_CompareWithRefsNumpy_dynamic_range, SelectLayerGPUTest, rangeNumpyCases, SelectLayerGPUTest::getTestCaseName);

// AutoBroadcastType: NONE
const std::vector<std::vector<InputShape>> inShapesDynamicNone = {
    {
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} },
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} },
        { {-1, -1, -1, -1}, {{10, 16, 20, 5}, {2, 1, 7, 1}} }
    },
    {
        { {{1, 10},       -1, {10, 20}, {1, 5}}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}} },
        { {     -1, {16, 16},       -1,     -1}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}} },
        { {     -1,       -1,       -1,     -1}, {{3, 16, 15, 5}, {1, 16, 10, 1}, {10, 16, 20, 5}} }
    }
};

const auto noneCases = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamicNone),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(op::AutoBroadcastType::NONE),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_select_CompareWithRefsNone_dynamic, SelectLayerGPUTest, noneCases, SelectLayerGPUTest::getTestCaseName);

} // namespace Select
} // namespace
} // namespace GPULayerTestsDefinitions
