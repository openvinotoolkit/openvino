// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/shape_of.hpp"
#include "shared_test_classes/single_layer/strided_slice.hpp"
#include <shared_test_classes/single_layer/eltwise.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ElementType, // Network precision
        TargetDevice, // Device name
        std::map<std::string, std::string> // Additional network configuration
> genImplKeyDynamicGPUTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::f16,
    ElementType::f32,
    ElementType::i32,
    ElementType::i64,
};

class GenlImplKeyDynamicGPUTest : public testing::WithParamInterface<genImplKeyDynamicGPUTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<genImplKeyDynamicGPUTestParamsSet>& obj) {
        genImplKeyDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        ElementType netType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
     void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
          inputs.clear();
          const auto& funcInputs = function->inputs();
          for (size_t i = 0; i < funcInputs.size(); ++i) {
              const auto& funcInput = funcInputs[i];
              ov::Tensor tensor;
              tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                               targetInputStaticShapes[i],
                                                               80,
                                                               0,
                                                               8);
              inputs.insert({funcInput.get_node_shared_ptr(), tensor});
          }
    }

    void SetUp() override {
        genImplKeyDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(inputShapes);
        const auto inShapeShapeOf = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : {inShapeShapeOf, inShapeElt})
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

        auto addOp1 = ngraph::builder::makeEltwise(params[1], params[1], ngraph::helpers::EltwiseTypes::ADD);
        addOp1->set_friendly_name("add1");

        auto shapeOfOp1 = std::make_shared<ngraph::opset3::ShapeOf>(addOp1, ElementType::i64);
        shapeOfOp1->set_friendly_name("shapeof1");

        std::vector<int> reduce_axes = {0};
        auto reduceAxesNode1 = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({1}), reduce_axes));
        auto reduceOp1 = ngraph::builder::makeReduce(shapeOfOp1, reduceAxesNode1, true, ngraph::helpers::ReductionType::Prod);
        reduceOp1->set_friendly_name("reduce1");

        std::vector<int64_t> shapePatternFill = {-1};
        auto reshapePatternComp1 = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64,
                                                                           ngraph::Shape{1}, shapePatternFill);
        auto concatOp1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reduceOp1, reshapePatternComp1}, 0);
        concatOp1->set_friendly_name("concat1");

        auto reshapeOp1 = std::make_shared<ngraph::opset1::Reshape>(addOp1, concatOp1, false);
        reshapeOp1->set_friendly_name("reshapeOp1");

        auto addOp2 = ngraph::builder::makeEltwise(params[1], params[1], ngraph::helpers::EltwiseTypes::ADD);
        addOp2->set_friendly_name("add2");

        auto shapeOfOp2 = std::make_shared<ngraph::opset3::ShapeOf>(addOp2, ElementType::i64);
        shapeOfOp2->set_friendly_name("shapeof2");

        auto reduceAxesNode2 = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({1}), reduce_axes));
        auto reduceOp2 = ngraph::builder::makeReduce(shapeOfOp2, reduceAxesNode2, true, ngraph::helpers::ReductionType::Prod);
        reduceOp2->set_friendly_name("reduce2");

        auto reshapePatternComp2 = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64,
                                                                           ngraph::Shape{1}, shapePatternFill);
        auto concatOp2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{reduceOp2, reshapePatternComp2}, 0);
        concatOp2->set_friendly_name("concat2");

        auto reshapeOp2 = std::make_shared<ngraph::opset1::Reshape>(addOp2, concatOp2, false);
        reshapeOp2->set_friendly_name("reshapeOp2");

        auto addOp3 = ngraph::builder::makeEltwise(reshapeOp1, reshapeOp2, ngraph::helpers::EltwiseTypes::ADD);
        addOp3->set_friendly_name("add3");

        auto shapeOf3 = std::make_shared<ngraph::opset3::ShapeOf>(addOp3, ElementType::i64);
        shapeOf3->set_friendly_name("shapeof3");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(shapeOf3)};
        function = std::make_shared<ngraph::Function>(results, params, "shapeof_out");
    }
};


TEST_P(GenlImplKeyDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    // 1D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic()}, {{30}, {40}, {50}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{3, 10}, {2, 20}, {25, 2}}}
    },
    // 2D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10}, {2, 20}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 1, 10}, {2, 10, 2}}}
    },
    // 3D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10, 4}, {1, 4, 12}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10, 4}, {2, 2, 12}}}
    },
    // 4D
    {
        // Input for ShapeOf
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{3, 1, 10, 4}, {2, 4, 23, 12}}},
        // Input for Add
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{30, 4}, {24, 92}}}
    }
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_impl_key, GenlImplKeyDynamicGPUTest,
                         testParams_smoke, GenlImplKeyDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
