// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/single_layer/strided_slice.hpp"
#include "shared_test_classes/single_layer/prior_box.hpp"
#include "shared_test_classes/single_layer/prior_box_clustered.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ngraph_functions/builders.hpp"
#include <string>
#include <openvino/pass/constant_folding.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

using ElementType = ov::element::Type_t;

namespace GPULayerTestsDefinitions {
typedef std::tuple<
        InputShape,
        InputShape,
        ElementType                // Net precision
> PriorBoxLayerGPUTestParamsSet;

class PriorBoxLayerGPUTest : public testing::WithParamInterface<PriorBoxLayerGPUTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PriorBoxLayerGPUTestParamsSet> obj) {
        InputShape input1Shape;
        InputShape input2Shape;
        ElementType netPrecision;
        std::tie(input1Shape, input1Shape, netPrecision) = obj.param;

        std::ostringstream result;
        result << "PriorBoxTest_";
        result << std::to_string(obj.index) << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "I1S=";
        result << CommonTestUtils::partialShape2str({input1Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input1Shape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")";
        result << "I2S=";
        result << CommonTestUtils::partialShape2str({input2Shape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : input2Shape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")";
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_GPU;

        auto netPrecision = ElementType::undefined;
        InputShape input1Shape;
        InputShape input2Shape;
        std::tie(input1Shape, input2Shape, netPrecision) = this->GetParam();

        init_input_shapes({input1Shape, input2Shape});

        inType = ov::element::Type(netPrecision);
        outType = ElementType::f32;

        auto beginInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto endInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {4});
        auto strideInput = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});

        auto functionParams = builder::makeDynamicParams(inType, inputDynamicShapes);
        auto staticParams = ngraph::builder::makeParams(inType, {{1, 3, 30, 30}, {1, 3, 224, 224}});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset3::Parameter>(staticParams));

        auto shapeOfOp1 = std::make_shared<opset3::ShapeOf>(paramOuts[0], element::i32);
        auto stridedSliceOp1 = ngraph::builder::makeStridedSlice(shapeOfOp1, beginInput, endInput, strideInput, element::i32,
                                                                {0}, {1}, {0}, {0}, {0});

        auto shapeOfOp2 = std::make_shared<opset3::ShapeOf>(paramOuts[1], element::i32);
        auto stridedSliceOp2 = ngraph::builder::makeStridedSlice(shapeOfOp2, beginInput, endInput, strideInput, element::i32,
                                                                {0}, {1}, {0}, {0}, {0});

        ngraph::op::v8::PriorBox::Attributes attributes;
        attributes.min_size = {64};
        attributes.max_size = {300};
        attributes.aspect_ratio = {2};
        attributes.variance = {0.1, 0.1, 0.2, 0.2};
        attributes.step = 16;
        attributes.offset = 0.5;
        attributes.clip = false;
        attributes.flip = true;
        attributes.scale_all_sizes = true;
        attributes.min_max_aspect_ratios_order = true;

        auto priorBoxOp = std::make_shared<ngraph::op::v8::PriorBox>(stridedSliceOp1, stridedSliceOp2, attributes);

        ov::pass::disable_constant_folding(priorBoxOp);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBoxOp)};
        function = std::make_shared <ngraph::Function>(results, staticParams, "PriorBoxFunction");
    }
};

TEST_P(PriorBoxLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
};

std::vector<ov::test::InputShape> inShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 30, 30 },
            }
        },
};
std::vector<ov::test::InputShape> imgShapesDynamic = {
        {
            {1, 3, -1, -1},
            {
                { 1, 3, 224, 224 },
            }
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_prior_box_dynamic,
    PriorBoxLayerGPUTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(imgShapesDynamic),
        ::testing::ValuesIn(netPrecisions)),
    PriorBoxLayerGPUTest::getTestCaseName);
} // namespace

} // namespace GPULayerTestsDefinitions
