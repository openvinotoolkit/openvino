// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

using InputShape = ov::test::InputShape;
using ElementType = ov::element::Type_t;

namespace ov {
namespace test {
typedef std::tuple<InputShape,
                   ElementType  // Net type
                   >
    ShapeOfLayerTestParams;

typedef std::tuple<ShapeOfLayerTestParams, CPUSpecificParams> ShapeOfLayerCPUTestParamsSet;

class ShapeOfLayerCPUTest : public testing::WithParamInterface<ShapeOfLayerCPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfLayerCPUTestParamsSet> obj) {
        ShapeOfLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        ElementType netPr;
        InputShape inputShape;

        std::tie(inputShape, netPr) = basicParamsSet;
        std::ostringstream result;
        result << "ShapeOfTest_";
        result << std::to_string(obj.index) << "_";
        result << "Prec=" << netPr << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ShapeOfLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        auto netPrecision = ElementType::dynamic;
        InputShape inputShape;
        std::tie(inputShape, netPrecision) = basicParamsSet;
        init_input_shapes({inputShape});

        inType = ov::element::Type(netPrecision);
        outType = ElementType::i32;
        selectedType = makeSelectedTypeStr("ref", inType);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(params.front(), ov::element::i32);

        function = makeNgraphFunction(netPrecision, params, shapeOf, "ShapeOf");
    }
};

TEST_P(ShapeOfLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ShapeOf");
}

namespace {

const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i32, ElementType::i8};

std::vector<ov::test::InputShape> inShapesDynamic3d = {{{-1, -1, -1}, {{8, 16, 4}, {8, 16, 3}, {8, 16, 2}}}};

std::vector<ov::test::InputShape> inShapesDynamic4d = {
    {{-1, -1, -1, -1}, {{8, 16, 3, 4}, {8, 16, 3, 3}, {8, 16, 3, 2}}},
};

std::vector<ov::test::InputShape> inShapesDynamic5d = {
    {{-1, -1, -1, -1, -1}, {{8, 16, 3, 2, 4}, {8, 16, 3, 2, 3}, {8, 16, 3, 2, 2}}}};
const auto params5dDynamic =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(inShapesDynamic5d), ::testing::ValuesIn(netPrecisions)),
                       ::testing::Values(emptyCPUSpec));

const auto params4dDynamic =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(inShapesDynamic4d), ::testing::ValuesIn(netPrecisions)),
                       ::testing::Values(emptyCPUSpec));

const auto params3dDynamic =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(inShapesDynamic3d), ::testing::ValuesIn(netPrecisions)),
                       ::testing::Values(emptyCPUSpec));

// We don't check static case, because of constant folding
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf3dDynamicLayoutTest,
                         ShapeOfLayerCPUTest,
                         params3dDynamic,
                         ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf4dDynamicLayoutTest,
                         ShapeOfLayerCPUTest,
                         params4dDynamic,
                         ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf5dDynamicLayoutTest,
                         ShapeOfLayerCPUTest,
                         params5dDynamic,
                         ShapeOfLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
