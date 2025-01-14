// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        InputShape, // input shapes
        ov::element::Type, // Network precision
        std::string, // Device name
        ov::test::utils::ActivationTypes, // Activation type
        std::vector<size_t>, //inShape
        std::vector<float> //constantValue
> shapeofActivationDynamicGPUTestParamsSet;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32,
};

class shapeofActivationDynamicGPUTest : public testing::WithParamInterface<shapeofActivationDynamicGPUTestParamsSet>,
                                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<shapeofActivationDynamicGPUTestParamsSet>& obj) {
        shapeofActivationDynamicGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        InputShape inputShape;
        ov::element::Type netType;
        std::string targetDevice;
        ov::test::utils::ActivationTypes activationType;
        std::vector<size_t> inShape;
        std::vector<float> constantValue;

        std::tie(inputShape, netType, targetDevice, activationType, inShape, constantValue) = basicParamsSet;
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
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 80;
            in_data.resolution = 8;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        shapeofActivationDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        InputShape inputShape;
        ov::element::Type netType;
        ov::test::utils::ActivationTypes activationType;
        std::vector<size_t> inShape;
        std::vector<float> constantValue;
        std::tie(inputShape, netType, targetDevice, activationType, inShape, constantValue) = basicParamsSet;

        init_input_shapes({inputShape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

        std::vector<int> shape_pattern = {0, 1, -1, 0};
        auto shapePatternsNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({4}), shape_pattern);
        auto reshapeOp = std::make_shared<ov::op::v1::Reshape>(params[0], shapePatternsNode, true);
        reshapeOp->set_friendly_name("reshape");

        auto shapeOfOp = std::make_shared<ov::op::v3::ShapeOf>(reshapeOp, ov::element::i32);
        shapeOfOp->set_friendly_name("shapeof");

        std::vector<int> indices = {0};
        auto indicesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape({1}), indices);
        std::vector<int> axis = {-1};
        auto axisNode = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape({1}), axis);
        auto gatherOp = std::make_shared<ov::op::v7::Gather>(shapeOfOp, indicesNode, axisNode, 0);
        gatherOp->set_friendly_name("gather");

        auto convertOp = std::make_shared<ov::op::v0::Convert>(gatherOp, ov::element::f32);
        convertOp->set_friendly_name("convert");

        auto activationOp = ov::test::utils::make_activation(convertOp,
                                                             netType,
                                                             activationType,
                                                             inShape,
                                                             constantValue);
        activationOp->set_friendly_name("sqrt");

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(activationOp)};
        function = std::make_shared<ov::Model>(results, params, "result");
    }
};


TEST_P(shapeofActivationDynamicGPUTest, Inference) {
    run();
}

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
                                                 ::testing::ValuesIn(model_types), // netprec
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::Values(ov::test::utils::ActivationTypes::Sqrt),
                                                 ::testing::Values(inShape_sqrt),
                                                 ::testing::Values(constantValue_sqrt));

// Activation type Sqrt should be supported in activation cpu_impl whic is selected after shapeOf
INSTANTIATE_TEST_SUITE_P(smoke_dynamic_shapeof_activation_sqrt, shapeofActivationDynamicGPUTest,
                         testParams_sqrt, shapeofActivationDynamicGPUTest::getTestCaseName);
} // namespace