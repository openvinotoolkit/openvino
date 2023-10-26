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
> shapeOfReshapeReduceDynamicGPUTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::f16,
    ElementType::f32,
    ElementType::i32,
    ElementType::i64,
};

class ShapeOfReshapeReduceDynamicGPUTest : public testing::WithParamInterface<shapeOfReshapeReduceDynamicGPUTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<shapeOfReshapeReduceDynamicGPUTestParamsSet>& obj) {
        shapeOfReshapeReduceDynamicGPUTestParamsSet basicParamsSet = obj.param;
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
        shapeOfReshapeReduceDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(inputShapes);
        const auto inShapeShapeOf = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
        }
        auto paramOuts = helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

        auto addOp = ngraph::builder::makeEltwise(paramOuts[1], paramOuts[1], ngraph::helpers::EltwiseTypes::ADD);
        addOp->set_friendly_name("add");

        auto shapeOfOp1 = std::make_shared<ngraph::opset3::ShapeOf>(paramOuts[0], ElementType::i64);
        shapeOfOp1->set_friendly_name("shapeof1");
        std::vector<int> reduce_axes = {0};
        auto reduceAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({1}), reduce_axes));
        auto reduceOp = ngraph::builder::makeReduce(shapeOfOp1, reduceAxesNode, true, ngraph::helpers::ReductionType::Prod);
        reduceOp->set_friendly_name("reduce");
        std::vector<int64_t> shapePatternFill = {-1};
        auto reshapePatternComp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64,
                                                                           ngraph::Shape{1}, shapePatternFill);
        auto concatOp = ngraph::builder::makeConcat({reduceOp, reshapePatternComp}, 0);
        concatOp->set_friendly_name("concat");

        auto reshapeOp = std::make_shared<ngraph::opset1::Reshape>(addOp, concatOp, false);

        auto shapeOf2 = std::make_shared<ngraph::opset3::ShapeOf>(reshapeOp, ElementType::i64);
        shapeOf2->set_friendly_name("shapeof2");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(shapeOf2)};
        function = std::make_shared<ngraph::Function>(results, params, "shapeof_out");
    }
};


TEST_P(ShapeOfReshapeReduceDynamicGPUTest, CompareWithRefs) {
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

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_shapeof_reshape, ShapeOfReshapeReduceDynamicGPUTest,
                         testParams_smoke, ShapeOfReshapeReduceDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
