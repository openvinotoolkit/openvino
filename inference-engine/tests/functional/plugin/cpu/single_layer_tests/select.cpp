// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using selectParams = std::tuple<
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>, // input shapes
    ngraph::op::AutoBroadcastSpec>;                                                        // broadcast

class SelectLayerCPUTest : public testing::WithParamInterface<selectParams>, public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<selectParams> obj) {
        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::tie(shapes, broadcast) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::partialShape2str(shapes.first) << "_";
        result << "TS=";
        for (const auto& shape : shapes.second) {
            result << "(";
            for (const auto& item : shape) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
            result << ")_";
        }
        result << "Broadcast=" << broadcast.m_type;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::tie(shapes, broadcast) = this->GetParam();

        for (size_t i = 0; i < shapes.second.size(); i++) {
            targetStaticShapes.push_back(shapes.second[i]);
        }
        inputDynamicShapes = shapes.first;

        ngraph::ParameterVector paramNodesVector;
        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::boolean, ngraph::Shape(targetStaticShapes[0][0]));
        paramNodesVector.push_back(paramNode);
        auto inType = ngraph::element::Type_t::f32;
        for (size_t i = 1; i < targetStaticShapes[0].size(); i++) {
            paramNode = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(targetStaticShapes[0][i]));
            paramNodesVector.push_back(paramNode);
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramNodesVector));

        auto select = ngraph::builder::makeSelect(paramOuts, broadcast);

        function = std::make_shared<ngraph::Function>(select, paramNodesVector, "SelectLayerCPUTest");
        functionRefs = ngraph::clone_function(*function);
    }
};

TEST_P(SelectLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamicNumpy = {
        {
            // dynamic
            {
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)}
            },

            // target
            {
                {{5, 1, 2, 1}, {8, 1, 9, 1, 1}, {5, 1, 2, 1}},
                {{1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1}},
                {{5, 9, 8, 7}, {21, 5, 9, 8, 7}, {1, 1, 1, 1}},
            }
        },
        {
            // dynamic
            {
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)}
            },

            // target
            {
                {{8, 1}, {2, 1, 1, 8, 1}, {9, 1, 1}},
                {{10, 5}, {7, 8, 3, 10, 5}, {3, 10, 5}},
                {{8, 7}, {1, 1, 1, 8, 1}, {1, 1, 7}},
            }
        },
        {
            // dynamic
            {
                {ngraph::Dimension(2, 8), ngraph::Dimension(3, 7), ngraph::Dimension(1, 10), ngraph::Dimension(1, 6), ngraph::Dimension(1, 10)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(1, 5), ngraph::Dimension(1, 11), ngraph::Dimension(5, 5), ngraph::Dimension(1, 8)}
            },

            // target
            {
                {{5, 4, 1, 1, 1}, {5, 1, 8, 1, 1}, {1, 1, 5, 1}},
                {{8, 5, 5, 5, 1}, {8, 1, 1, 1, 8}, {5, 5, 5, 8}},
                {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {3, 4, 5, 6}},
            }
        },
        {
            // dynamic
            {
                {ngraph::Dimension(1, 10)},
                {ngraph::Dimension(1, 15), ngraph::Dimension(2, 7), ngraph::Dimension(1, 6), ngraph::Dimension(5, 12), ngraph::Dimension(1, 20)},
                {ngraph::Dimension(2, 10), ngraph::Dimension(1, 16)}
            },

            // target
            {
                {{4}, {8, 5, 6, 6, 1}, {6, 4}},
                {{10}, {15, 7, 6, 10, 10}, {10, 10}},
                {{1}, {2, 5, 4, 5, 3}, {5, 1}},
            }
        }
};

const auto numpyCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesDynamicNumpy),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY)
);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_dynamic, SelectLayerCPUTest, numpyCases, SelectLayerCPUTest::getTestCaseName);

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamicNone = {
        {
            // dynamic
            {
                {ngraph::Dimension(1, 10), ngraph::Dimension(-1, -1), ngraph::Dimension(10, 20), ngraph::Dimension(1, 5)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(16, 16), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)},
                {ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1), ngraph::Dimension(-1, -1)}
            },

            // target
            {
                {{3, 16, 15, 5}, {3, 16, 15, 5}, {3, 16, 15, 5}},
                {{1, 16, 10, 1}, {1, 16, 10, 1}, {1, 16, 10, 1}},
                {{10, 16, 20, 5}, {10, 16, 20, 5}, {10, 16, 20, 5}}
            }
        }
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(inShapesDynamicNone),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE)
);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNone_dynamic, SelectLayerCPUTest, noneCases, SelectLayerCPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions