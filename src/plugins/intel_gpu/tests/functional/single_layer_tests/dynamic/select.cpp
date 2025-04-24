// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
    std::vector<InputShape>,   // input shapes
    ov::element::Type,         // presion of 'then' and 'else' of inputs
    ov::op::AutoBroadcastSpec, // broadcast spec
    std::string                // device name
> SelectLayerTestParamSet;

class SelectLayerGPUTest : public testing::WithParamInterface<SelectLayerTestParamSet>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectLayerTestParamSet>& obj) {
        std::vector<InputShape> inshapes;
        ov::element::Type model_type;
        ov::op::AutoBroadcastSpec broadcast;
        std::string targetDevice;
        std::tie(inshapes, model_type, broadcast, targetDevice) = obj.param;

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
        result << "Precision=" << model_type << "_";
        result << "Broadcast=" << broadcast.m_type << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inshapes;
        ov::element::Type model_type;
        ov::op::AutoBroadcastSpec broadcast;
        std::tie(inshapes, model_type, broadcast, targetDevice) = this->GetParam();

        init_input_shapes(inshapes);

        ov::ParameterVector params = {
            std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[0]),
            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
            std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2]),
        };

        auto select = std::make_shared<ov::op::v1::Select>(params[0], params[1], params[2], broadcast);

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "SelectLayerGPUTest");
        };
        function = makeFunction(params, select);
    }
};

TEST_P(SelectLayerGPUTest, Inference) {
   run();
}

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

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
    {
        { {            -1}, {{         130048}} },
        { {        -1, -1}, {{      2, 130048}} },
        { {        -1, -1}, {{      2, 130048}} }
    },
};

const auto numpyCases = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamicNumpy),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
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
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
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
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::op::AutoBroadcastType::NONE),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_select_CompareWithRefsNone_dynamic, SelectLayerGPUTest, noneCases, SelectLayerGPUTest::getTestCaseName);
} // namespace
