// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/activation.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

using InputShape = ov::test::InputShape;
using ElementType = ov::element::Type_t;

namespace ov {
namespace test {

//   ┌────────┐
//   │ Param  │
//   └───┬────┘
//       │
//       │
//       │
//   ┌───┴────┐ To simulate different layouts
//   │ Eltwise│ ◄─────────────────────────────
//   └───┬────┘
//       │     No Reorders are expected
//       │    ◄───────────────────────────
//       │
//   ┌───┴────┐
//   │ShapeOf │
//   └───┬────┘
//       │
//       │
//       │
//   ┌───┴────┐
//   │ Output │
//   └────────┘

typedef std::tuple<
        InputShape,
        ElementType                // Net precision
> ShapeOfAnyLayoutParams;

typedef std::tuple<
        ShapeOfAnyLayoutParams,
        CPUSpecificParams
> ShapeOfAnyLayoutCPUTestParamsSet;

class ShapeOfAnyLayoutCPUTest : public testing::WithParamInterface<ShapeOfAnyLayoutCPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfAnyLayoutCPUTestParamsSet> obj) {
        ShapeOfAnyLayoutParams basicParamsSet;
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

        ShapeOfAnyLayoutParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::vector<cpu_memory_format_t> eltwiseInFmts, eltwiseOutFmts;
        std::tie(eltwiseInFmts, eltwiseOutFmts, priority, selectedType) = cpuParams;

        auto netPrecision = ElementType::undefined;
        InputShape inputShape;
        std::tie(inputShape, netPrecision) = basicParamsSet;
        init_input_shapes({inputShape});

        inType = ov::element::Type(netPrecision);
        outType = ElementType::i32;
        selectedType = makeSelectedTypeStr("ref", inType);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));

        //make a stub eltwise node to enforce layout, since ShapeOf just mimic any input layout
        auto eltwise = utils::make_activation(params[0], inType, ov::test::utils::ActivationTypes::Relu);
        eltwise->get_rt_info() = makeCPUInfo(eltwiseInFmts, eltwiseOutFmts, {});

        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(eltwise, ov::element::i32);

        function = makeNgraphFunction(netPrecision, params, shapeOf, "ShapeOf");
    }
};

TEST_P(ShapeOfAnyLayoutCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ShapeOf");
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 1);
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> getCpuInfoForDimsCount(const size_t dimsCount = 3) {
    std::vector<CPUSpecificParams> resCPUParams;
    const bool avx512_target = with_cpu_x86_avx512f();

    if (dimsCount == 5) {
        auto blocked_format = avx512_target ? nCdhw16c : nCdhw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {blocked_format}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {ndhwc}, {}, {}});
    } else if (dimsCount == 4) {
        auto blocked_format = avx512_target ? nChw16c : nChw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {blocked_format}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {nhwc}, {}, {}});
    } else {
        auto blocked_format = avx512_target ? nCw16c : nCw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {blocked_format}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{acb}, {acb}, {}, {}});
    }

    return filterCPUSpecificParams(resCPUParams);
}

const std::vector<ElementType> netPrecisions = {
        ElementType::f32
};

std::vector<ov::test::InputShape> inShapesDynamic3d = {
        {
            {-1, 16, -1},
            {
                { 8, 16, 4 },
                { 8, 16, 3 },
                { 8, 16, 2 }
            }
        }
};

std::vector<ov::test::InputShape> inShapesDynamic4d = {
        {
            {-1, 16, -1, -1},
            {
                { 8, 16, 3, 4 },
                { 8, 16, 3, 3 },
                { 8, 16, 3, 2 }
            }
        },
};

std::vector<ov::test::InputShape> inShapesDynamic5d = {
        {
            { -1, 16, -1, -1, -1 },
            {
                { 8, 16, 3, 2, 4 },
                { 8, 16, 3, 2, 3 },
                { 8, 16, 3, 2, 2 }
            }
        }
};
const auto params5dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapesDynamic5d),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(getCpuInfoForDimsCount(5)));

const auto params4dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapesDynamic4d),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(getCpuInfoForDimsCount(4)));

const auto params3dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapesDynamic3d),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(getCpuInfoForDimsCount(3)));

// We don't check static case, because of constant folding
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf3dAnyLayoutTest, ShapeOfAnyLayoutCPUTest,
                         params3dDynamic, ShapeOfAnyLayoutCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf4dAnyLayoutTest, ShapeOfAnyLayoutCPUTest,
                         params4dDynamic, ShapeOfAnyLayoutCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf5dAnyLayoutTest, ShapeOfAnyLayoutCPUTest,
                         params5dDynamic, ShapeOfAnyLayoutCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
