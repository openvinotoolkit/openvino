// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using logSoftmaxLayerTestParams = std::tuple<std::vector<InputShape>,  // inputShape
                                             ov::element::Type,        // netPrecision
                                             int64_t>;                 // axis

class LogSoftmaxLayerCPUTest
        : public testing::WithParamInterface<logSoftmaxLayerTestParams>,
          public SubgraphBaseTest,
          public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<logSoftmaxLayerTestParams> obj) {
        std::vector<InputShape> inputShapes;
        ov::element::Type netPrecision;
        int64_t axis;
        std::tie(inputShapes, netPrecision, axis) = obj.param;

        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << ov::test::utils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto &shape : inputShapes) {
            for (const auto &item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "netPRC=" << netPrecision.to_string();
        result << "Axis=" << axis;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        ov::element::Type netPrecision;
        int64_t axis;
        std::tie(inputShapes, netPrecision, axis) = this->GetParam();

        auto ngPrc = netPrecision;
        inType = outType = ngPrc;

        selectedType = std::string("unknown_") + netPrecision.to_string();
        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes.front())};
        const auto logSoftmax = std::make_shared<ov::op::v5::LogSoftmax>(params[0], axis);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(logSoftmax)};
        function = std::make_shared<ov::Model>(results, params, "logSoftmax");
    }
};

TEST_P(LogSoftmaxLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "logSoftmax");
}

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<std::vector<InputShape>> inputShapes2D = {
        {
                {{{-1, -1}, {{1, 100}, {100, 1}, {10, 10}}}},
                {{{-1, {1}}, {{1, 1}, {100, 1}, {10, 1}}}}
        }
};

const std::vector<int64_t> axis2D = {
        -2, -1, 0, 1
};

const auto params2D = testing::Combine(
        testing::ValuesIn(inputShapes2D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis2D));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax2D_dynamic, LogSoftmaxLayerCPUTest, params2D,
                         LogSoftmaxLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapes4D = {
        {
                {{{-1, -1, -1, -1}, {{1, 100, 1, 1}, {1, 3, 4, 3}, {2, 3, 4, 5}}}},
                {{{{1, 2}, -1, {1, 5}, -1}, {{1, 100, 1, 1}, {1, 3, 5, 3}, {2, 3, 4, 5}}}}
        }
};

const std::vector<int64_t> axis4D = {
        -4, -3, -2, -1, 0, 1, 2, 3
};

const auto params4D = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis4D));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax4D_dynamic, LogSoftmaxLayerCPUTest, params4D,
                         LogSoftmaxLayerCPUTest::getTestCaseName);

} // namespace
}  // namespace test
}  // namespace ov
