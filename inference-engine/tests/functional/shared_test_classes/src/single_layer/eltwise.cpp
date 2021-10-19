// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string EltwiseLayerTest::getTestCaseName(const testing::TestParamInfo<EltwiseTestParams>& obj) {
    std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
    ElementType netType;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseOpType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, eltwiseOpType, secondaryInputType, opType, netType, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::partialShape2str(shapes.first) << "_";
    results << "TS=";
    for (const auto& shape : shapes.second) {
        results << "(";
        for (const auto& item : shape) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
        results << ")_";
    }
    results << "eltwiseOpType=" << eltwiseOpType << "_";
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "opType=" << opType << "_";
    results << "NetType=" << netType << "_";
    results << "trgDev=" << targetName;
    return results.str();
}

void EltwiseLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto opType = std::get<1>(GetParam());
    const auto& params = function->get_parameters();
    for (int i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        ov::runtime::Tensor tensor;
        bool isReal = param->get_element_type().is_real();
        switch (opType) {
            case ngraph::helpers::EltwiseTypes::POWER:
            case ngraph::helpers::EltwiseTypes::MOD:
            case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
                tensor = isReal ?
                        ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i], 2, 2, 128) :
                        ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i], 4, 2);
                break;
            case ngraph::helpers::EltwiseTypes::DIVIDE:
                tensor = isReal ?
                         ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i], 2, 2, 128) :
                         ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i], 100, 101);
                break;
            case ngraph::helpers::EltwiseTypes::ERF:
                tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i], 6, -3);
                break;
            default:
                tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i]);
                break;
        }
        inputs.insert({param->get_friendly_name(), tensor});
    }
}

void EltwiseLayerTest::transformInputShapesAccordingEltwise(const ov::PartialShape& secondInputShape) {
    // propagate shapes in case 1 shape is defined
    if (inputDynamicShapes.size() == 1) {
        inputDynamicShapes.push_back(inputDynamicShapes.front());
        for (auto& staticShape : targetStaticShapes) {
            staticShape.push_back(staticShape.front());
        }
    }
    ASSERT_EQ(inputDynamicShapes.size(), 2) << "Incorrect inputs number!";
    if (secondInputShape.get_shape() == ov::Shape{1}) {
        inputDynamicShapes[1] = secondInputShape;
        for (auto& staticShape : targetStaticShapes) {
            staticShape[1] = secondInputShape.get_shape();
        }
    }
}

void EltwiseLayerTest::SetUp() {
    InputShapes shapes;
    ElementType netType;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseType;
    Config additional_config;
    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, targetDevice, configuration) =
        this->GetParam();

    init_input_shapes(shapes);

    auto parameters = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.front()});

    ov::PartialShape shape_input_secondary;
    switch (opType) {
        case CommonTestUtils::OpType::SCALAR: {
            shape_input_secondary = {1};
            break;
        }
        case CommonTestUtils::OpType::VECTOR:
            shape_input_secondary = inputDynamicShapes.back();
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }
    // To propagate shape_input_secondary just in static case because all shapes are defined in dynamic scenarion
    if (shape_input_secondary.is_static()) {
        transformInputShapesAccordingEltwise(shape_input_secondary);
    }

    std::shared_ptr<ngraph::Node> secondaryInput;
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        secondaryInput = ngraph::builder::makeDynamicParams(netType, {shape_input_secondary}).front();
        parameters.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    } else {
        ov::Shape shape = shape_input_secondary.get_shape();
        switch (eltwiseType) {
            case ngraph::helpers::EltwiseTypes::DIVIDE:
            case ngraph::helpers::EltwiseTypes::MOD:
            case ngraph::helpers::EltwiseTypes::FLOOR_MOD: {
                std::vector<float> data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape), 10, 2);
                secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
                break;
            }
            case ngraph::helpers::EltwiseTypes::POWER:
                secondaryInput = ngraph::builder::makeConstant<float>(netType, shape, {}, true, 3);
                break;
            default:
                secondaryInput = ngraph::builder::makeConstant<float>(netType, shape, {}, true);
        }
    }

    parameters[0]->set_friendly_name("param0");
    secondaryInput->set_friendly_name("param1");

    auto eltwise = ngraph::builder::makeEltwise(parameters[0], secondaryInput, eltwiseType);
    function = std::make_shared<ngraph::Function>(eltwise, parameters, "Eltwise");
}
} // namespace subgraph
} // namespace test
} // namespace ov
