// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

using InputShape = ov::test::InputShape;
using ElementType = ov::element::Type_t;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        InputShape,
        ElementType                // Net precision
> ShapeOfLayerTestParams;

typedef std::tuple<
        ShapeOfLayerTestParams,
        CPUSpecificParams
> ShapeOfLayerCPUTestParamsSet;

class ShapeOfLayerCPUTest : public testing::WithParamInterface<ShapeOfLayerCPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::ShapeOfLayerTestParams basicParamsSet;
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

        CPULayerTestsDefinitions::ShapeOfLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

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
        auto eltwise = ngraph::builder::makeActivation(params[0], inType, ov::test::utils::ActivationTypes::Relu);
        eltwise->get_rt_info() = makeCPUInfo(inFmts, inFmts, {});

        auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(eltwise, ngraph::element::i32);

        function = makeNgraphFunction(netPrecision, params, shapeOf, "ShapeOf");
    }
};

TEST_P(ShapeOfLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ShapeOf");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> getCpuInfoForDimsCount(const size_t dimsCount = 3) {
    std::vector<CPUSpecificParams> resCPUParams;
    const bool avx512_target = with_cpu_x86_avx512f();

    if (dimsCount == 5) {
        auto blocked_format = avx512_target ? nCdhw16c : nCdhw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {x}, {}, {}});
    } else if (dimsCount == 4) {
        auto blocked_format = avx512_target ? nChw16c : nChw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {x}, {}, {}});
    } else {
        auto blocked_format = avx512_target ? nCw16c : nCw8c;
        resCPUParams.push_back(CPUSpecificParams{{blocked_format}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{abc}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{acb}, {x}, {}, {}});
    }

    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i8
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
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf3dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params3dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf4dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params4dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf5dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params5dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
