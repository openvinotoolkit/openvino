// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using cumSumParams = std::tuple<ov::element::Type,  // data precision
                                InputShape,         // input shape
                                std::int64_t,       // axis
                                bool,               // exclusive
                                bool>;              // reverse

class CumSumLayerCPUTest : public testing::WithParamInterface<cumSumParams>,
                           public SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<cumSumParams> obj) {
        ov::element::Type inputPrecision;
        InputShape shapes;
        std::int64_t axis;
        bool exclusive;
        bool reverse;
        std::tie(inputPrecision, shapes, axis, exclusive, reverse) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "Prc=" << inputPrecision << "_";
        results << "Axis=" << axis << "_" << (exclusive ? "exclusive" : "") << "_" << (reverse ? "reverse" : "");
        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape shapes;
        std::int64_t axis;
        bool exclusive;
        bool reverse;
        std::tie(inType, shapes, axis, exclusive, reverse) = this->GetParam();
        if (inType == ElementType::bf16)
            rel_threshold = 0.05f;

        selectedType = makeSelectedTypeStr("ref_any", inType);
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto axisNode =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, std::vector<int64_t>{axis})->output(0);
        auto cumSum = std::make_shared<ov::op::v0::CumSum>(params[0], axisNode, exclusive, reverse);

        function = std::make_shared<ov::Model>(ov::NodeVector{cumSum}, params, "CumSumLayerCPUTest");
        functionRefs = function->clone();
    }
};

TEST_P(CumSumLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CumSum");
}

const std::vector<ov::element::Type> inputPrecision = {ov::element::i8, ov::element::bf16, ov::element::f32};

const std::vector<int64_t> axes = {0, 1, 2, 3, 4, 5, 6};
const std::vector<int64_t> negativeAxes = {-1, -2, -3, -4, -5, -6};

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse = {true, false};

const std::vector<InputShape> inShapes = {
    {{-1}, {{16}, {18}, {12}}},

    {{-1, -1}, {{9, 15}, {18, 12}, {12, 12}}},

    {{-1, -1, -1}, {{16, 10, 12}, {18, 12, 10}, {12, 18, 10}}},

    {{-1, -1, -1, -1}, {{18, 20, 14, 12}, {19, 20, 14, 12}, {20, 22, 23, 25}}},

    {{-1, -1, -1, -1, -1}, {{2, 4, 6, 2, 4}, {3, 5, 6, 3, 5}, {1, 4, 2, 6, 8}}},

    {{-1, -1, -1, -1, -1, -1}, {{2, 4, 6, 2, 4, 2}, {3, 5, 6, 3, 5, 3}, {1, 4, 2, 6, 8, 1}}},

    {{{-1, -1, -1, -1, -1, -1, -1}}, {{2, 4, 6, 2, 4, 2, 4}, {3, 5, 6, 3, 5, 3, 5}, {1, 4, 2, 6, 8, 1, 4}}},

    {{{2, 5}, {3, 7}, {4, 8}, {5, 7}, {2, 5}, {3, 7}, {1, 2}},
     {{2, 4, 6, 5, 4, 3, 1}, {3, 5, 6, 6, 5, 3, 1}, {5, 7, 4, 6, 3, 7, 2}}},

    {{{2, 5}, -1, {4, 8}, -1, -1, {3, 7}, -1}, {{2, 4, 6, 5, 4, 3, 1}, {3, 5, 6, 6, 5, 3, 1}, {5, 7, 4, 6, 3, 7, 2}}}};

const auto testCasesAxis_0 = ::testing::Combine(::testing::ValuesIn(inputPrecision),
                                                ::testing::ValuesIn(inShapes),
                                                ::testing::Values(axes[0]),
                                                ::testing::ValuesIn(exclusive),
                                                ::testing::ValuesIn(reverse));

const auto testCasesAxis_1 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 1, inShapes.end())),
                       ::testing::Values(axes[1]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_2 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 2, inShapes.end())),
                       ::testing::Values(axes[2]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_3 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 3, inShapes.end())),
                       ::testing::Values(axes[3]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_4 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 4, inShapes.end())),
                       ::testing::Values(axes[4]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_5 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 5, inShapes.end())),
                       ::testing::Values(axes[5]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_6 =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 6, inShapes.end())),
                       ::testing::Values(axes[6]),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

const auto testCasesAxis_negative =
    ::testing::Combine(::testing::ValuesIn(inputPrecision),
                       ::testing::ValuesIn(std::vector<InputShape>(inShapes.begin() + 6, inShapes.end())),
                       ::testing::ValuesIn(negativeAxes),
                       ::testing::ValuesIn(exclusive),
                       ::testing::ValuesIn(reverse));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_0,
                         CumSumLayerCPUTest,
                         testCasesAxis_0,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_1,
                         CumSumLayerCPUTest,
                         testCasesAxis_1,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_2,
                         CumSumLayerCPUTest,
                         testCasesAxis_2,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_3,
                         CumSumLayerCPUTest,
                         testCasesAxis_3,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_4,
                         CumSumLayerCPUTest,
                         testCasesAxis_4,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_5,
                         CumSumLayerCPUTest,
                         testCasesAxis_5,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_axis_6,
                         CumSumLayerCPUTest,
                         testCasesAxis_6,
                         CumSumLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNumpy_negative_axes,
                         CumSumLayerCPUTest,
                         testCasesAxis_negative,
                         CumSumLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
