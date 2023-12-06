// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
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

        outType = ElementType::i32;

        ov::ParameterVector functionParams;
        for (auto&& shape : inputDynamicShapes)
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

        auto shapeOfOp = std::make_shared<opset3::ShapeOf>(functionParams[0], element::i32);

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

using ShapeOfParams = typename std::tuple<
        InputShape,                     // Shape
        InferenceEngine::Precision,     // Precision
        LayerTestsUtils::TargetDevice   // Device name
>;

class ShapeOfDynamicInputGPUTest : public testing::WithParamInterface<ShapeOfParams>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ShapeOfParams>& obj) {
        InputShape inputShapes;
        InferenceEngine::Precision dataPrc;
        std::string targetDevice;

        std::tie(inputShapes, dataPrc, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "netPRC=" << dataPrc << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        InputShape inputShapes;
        InferenceEngine::Precision dataPrc;
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::tie(inputShapes, dataPrc, targetDevice) = GetParam();

        init_input_shapes({inputShapes});

        InferenceEngine::PreProcessInfo pre_process_info;
        pre_process_info.setVariant(InferenceEngine::MeanVariant::MEAN_VALUE);

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrc);

        auto input = std::make_shared<ngraph::opset9::Parameter>(prc, inputShapes.first);
        input->get_output_tensor(0).get_rt_info()["ie_legacy_preproc"] = pre_process_info;
        input->set_friendly_name("input_data");

        auto shape_of_01 = std::make_shared<ngraph::opset9::ShapeOf>(input);
        shape_of_01->set_friendly_name("shape_of_01");

        auto shape_of_02 = std::make_shared<ngraph::opset9::ShapeOf>(shape_of_01);
        shape_of_02->set_friendly_name("shape_of_02");

        auto result = std::make_shared<ngraph::opset1::Result>(shape_of_02);
        result->set_friendly_name("outer_result");

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{result}, ngraph::ParameterVector{input});
        function->set_friendly_name("shape_of_test");
    }
};

TEST_P(ShapeOfDynamicInputGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

const std::vector<ov::test::InputShape> dynamicInputShapes = {
    ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{4, 1, 1, 64, 32}, {6, 1, 1, 8, 4}, {8, 1, 1, 24, 16}}),
};

const std::vector<InferenceEngine::Precision> dynamicInputPrec = {
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ShapeOfDynamicInputGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes),                          // input shapes
                    testing::ValuesIn(dynamicInputPrec),                               // network precision
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),     // device type
                ShapeOfDynamicInputGPUTest::getTestCaseName);

} // namespace GPULayerTestsDefinitions
