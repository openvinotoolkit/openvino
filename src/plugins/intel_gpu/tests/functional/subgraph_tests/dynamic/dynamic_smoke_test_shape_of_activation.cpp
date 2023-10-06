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
#include "shared_test_classes/single_layer/reshape.hpp"
#include "shared_test_classes/single_layer/gather.hpp"
#include "shared_test_classes/single_layer/activation.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        InputShape, // input shapes
        ElementType, // Network precision
        TargetDevice, // Device name
        ngraph::helpers::ActivationTypes, // Activation type
        std::vector<size_t>, //inShape
        std::vector<float>, //constantValue
        std::map<std::string, std::string> // Additional network configuration
> shapeofActivationDynamicGPUTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::f16,
    ElementType::f32,
};

class shapeofActivationDynamicGPUTest : public testing::WithParamInterface<shapeofActivationDynamicGPUTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<shapeofActivationDynamicGPUTestParamsSet>& obj) {
        shapeofActivationDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        InputShape inputShape;
        ElementType netType;
        TargetDevice targetDevice;
        ngraph::helpers::ActivationTypes activationType;
        std::vector<size_t> inShape;
        std::vector<float> constantValue;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShape, netType, targetDevice, activationType, inShape, constantValue, additionalConfig) = basicParamsSet;
        result << "IS=";
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        for (const auto& actual_shape : inputShape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice << "_";
        result << "activatioinType=" << activationType << "_";
        result << "inShape=" << ov::test::utils::vec2str(inShape) << "_";
        result << "constantValue=" << ov::test::utils::vec2str(constantValue);
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
        shapeofActivationDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        InputShape inputShape;
        ElementType netType;
        ngraph::helpers::ActivationTypes activationType;
        std::vector<size_t> inShape;
        std::vector<float> constantValue;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShape, netType, targetDevice, activationType, inShape, constantValue, additionalConfig) = basicParamsSet;

        init_input_shapes({inputShape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
        }
        auto paramOuts = helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

        std::vector<int> shape_pattern = {0, 1, -1, 0};
        auto shapePatternsNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({4}), shape_pattern));
        auto reshapeOp = std::make_shared<ngraph::opset1::Reshape>(paramOuts[0], shapePatternsNode, true);
        reshapeOp->set_friendly_name("reshape");

        auto shapeOfOp = std::make_shared<ngraph::opset3::ShapeOf>(reshapeOp, ElementType::i32);
        shapeOfOp->set_friendly_name("shapeof");

        std::vector<int> indices = {0};
        auto indicesNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ngraph::Shape({1}), indices));
        std::vector<int> axis = {-1};
        auto axisNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ngraph::Shape({1}), axis));
        auto gatherOp = std::make_shared<ov::op::v7::Gather>(shapeOfOp, indicesNode, axisNode, 0);
        gatherOp->set_friendly_name("gather");

        auto convertOp = ngraph::builder::makeConversion(gatherOp, ElementType::f32,
                                                         ngraph::helpers::ConversionTypes::CONVERT);
        convertOp->set_friendly_name("convert");

        auto activationOp = ngraph::builder::makeActivation(convertOp,
                                                            netType,
                                                            activationType,
                                                            inShape,
                                                            constantValue);
        activationOp->set_friendly_name("sqrt");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(activationOp)};
        function = std::make_shared<ngraph::Function>(results, params, "result");
    }
};


TEST_P(shapeofActivationDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
std::vector<ov::test::InputShape> inShapesDynamic4d = {
        {
            {-1, -1, 1, 64},
            {
                { 1, 3136, 1, 64 },
                { 1, 49, 1, 64 },
                { 2, 49, 1, 64 }
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

std::vector<size_t> inShape_sqrt = {};
std::vector<float> constantValue_sqrt = {};

const auto testParams_sqrt = ::testing::Combine(::testing::ValuesIn(inShapesDynamic4d),
                                                 ::testing::ValuesIn(netPrecisions), // netprec
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::Values(ngraph::helpers::ActivationTypes::Sqrt),
                                                 ::testing::Values(inShape_sqrt),
                                                 ::testing::Values(constantValue_sqrt),
                                                 ::testing::Values(emptyAdditionalConfig));

// Activation type Sqrt should be supported in activation cpu_impl whic is selected after shapeOf
INSTANTIATE_TEST_SUITE_P(smoke_dynamic_shapeof_activation_sqrt, shapeofActivationDynamicGPUTest,
                         testParams_sqrt, shapeofActivationDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
