// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

using ElementType = ov::element::Type_t;

namespace GPULayerTestsDefinitions {
typedef std::tuple<
        InputShape,
        ElementType                // Net precision
> ShapeOfLayerGPUTestParamsSet;

class ShapeOfLayerGPUTest : public testing::WithParamInterface<ShapeOfLayerGPUTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfLayerGPUTestParamsSet> obj) {
        InputShape inputShape;
        ElementType netPrecision;
        std::tie(inputShape, netPrecision) = obj.param;

        std::ostringstream result;
        result << "ShapeOfTest_";
        result << std::to_string(obj.index) << "_";
        result << "netPrec=" << netPrecision << "_";
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
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto netPrecision = ElementType::undefined;
        InputShape inputShape;
        std::tie(inputShape, netPrecision) = this->GetParam();

        init_input_shapes({inputShape});

        inType = ov::element::Type(netPrecision);
        outType = ElementType::i32;

        auto functionParams = builder::makeDynamicParams(inType, inputDynamicShapes);
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset3::Parameter>(functionParams));
        auto shapeOfOp = std::make_shared<opset3::ShapeOf>(paramOuts[0], element::i32);

        auto makeFunction = [](ParameterVector &params, const std::shared_ptr<Node> &lastNode) {
            ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<opset1::Result>(lastNode->output(i)));

            return std::make_shared<Function>(results, params, "ShapeOfLayerGPUTest");
        };

        function = makeFunction(functionParams, shapeOfOp);
    }
};

TEST_P(ShapeOfLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> netPrecisions = {
        ElementType::i32,
};

// We don't check static case, because of constant folding

// ==============================================================================
// 3D
std::vector<ov::test::InputShape> inShapesDynamic3d = {
        {
            {-1, -1, -1},
            {
                { 8, 5, 4 },
                { 8, 5, 3 },
                { 8, 5, 2 }
            }
        },
        {
            {-1, -1, -1},
            {
                { 1, 2, 4 },
                { 1, 2, 3 },
                { 1, 2, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_3d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic3d),
        ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<Shape> inShapesStatic3d = {
    { 8, 5, 4 },
    { 8, 5, 3 },
    { 8, 5, 2 },
    { 1, 2, 4 },
    { 1, 2, 3 },
    { 1, 2, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_3d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
            ::testing::ValuesIn(static_shapes_to_test_representation(inShapesStatic3d)),
            ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

// ==============================================================================
// 4D
std::vector<ov::test::InputShape> inShapesDynamic4d = {
        {
            {-1, -1, -1, -1},
            {
                { 8, 5, 3, 4 },
                { 8, 5, 3, 3 },
                { 8, 5, 3, 2 }
            }
        },
        {
            {-1, -1, -1, -1},
            {
                { 1, 2, 3, 4 },
                { 1, 2, 3, 3 },
                { 1, 2, 3, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_4d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic4d),
        ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<Shape> inShapesStatic4d = {
    { 8, 5, 3, 4 },
    { 8, 5, 3, 3 },
    { 8, 5, 3, 2 },
    { 1, 2, 3, 4 },
    { 1, 2, 3, 3 },
    { 1, 2, 3, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_4d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapesStatic4d)),
        ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

// ==============================================================================
// 5D
std::vector<ov::test::InputShape> inShapesDynamic5d = {
        {
            { -1, -1, -1, -1, -1 },
            {
                { 8, 5, 3, 2, 4 },
                { 8, 5, 3, 2, 3 },
                { 8, 5, 3, 2, 2 }
            }
        },
        {
            {-1, -1, -1, -1, -1},
            {
                { 1, 2, 3, 4, 4 },
                { 1, 2, 3, 4, 3 },
                { 1, 2, 3, 4, 2 }
            }
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_5d_compareWithRefs_dynamic,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic5d),
        ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

std::vector<Shape> inShapesStatic5d = {
    { 8, 5, 3, 2, 4 },
    { 8, 5, 3, 2, 3 },
    { 8, 5, 3, 2, 2 },
    { 1, 2, 3, 4, 4 },
    { 1, 2, 3, 4, 3 },
    { 1, 2, 3, 4, 2 }
};
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_5d_compareWithRefs_static,
    ShapeOfLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapesStatic5d)),
        ::testing::ValuesIn(netPrecisions)),
    ShapeOfLayerGPUTest::getTestCaseName);

} // namespace

} // namespace GPULayerTestsDefinitions
