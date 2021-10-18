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

void EltwiseLayerTest::SetUp() {
    InputShapes shapes;
    ElementType netType;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseType;
    Config additional_config;
    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, targetDevice, additional_config) =
        this->GetParam();

    init_input_shapes(shapes);

    ngraph::Shape inputShape1 = targetStaticShapes.front().front(), inputShape2 = targetStaticShapes.front().back();

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(netType, {inputShape1});

    std::vector<size_t> shape_input_secondary;
    switch (opType) {
        case CommonTestUtils::OpType::SCALAR: {
            shape_input_secondary = std::vector<size_t>({1});
            break;
        }
        case CommonTestUtils::OpType::VECTOR:
            shape_input_secondary = inputShape2;
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }

    std::shared_ptr<ngraph::Node> secondaryInput;
    if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
        eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD ||
        eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
        std::vector<float> data(ngraph::shape_size(shape_input_secondary));
        data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
        secondaryInput = ngraph::builder::makeConstant(netType, shape_input_secondary, data);
    } else if (eltwiseType == ngraph::helpers::EltwiseTypes::POWER && secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
        // to avoid floating point overflow on some platforms, let's fill the constant with small numbers.
        secondaryInput = ngraph::builder::makeConstant<float>(netType, shape_input_secondary, {}, true, 3);
    } else {
        secondaryInput = ngraph::builder::makeInputLayer(netType, secondaryInputType, shape_input_secondary);
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        }
    }
    input[0]->set_friendly_name("param0");
    secondaryInput->set_friendly_name("param1");

    auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);
    function = std::make_shared<ngraph::Function>(eltwise, input, "Eltwise");
    // w/a: to propagate 1 input shape for other input
    for (auto& staticShape : targetStaticShapes) {
        if (function->get_parameters().size() > staticShape.size()) {
            for (size_t i = 0; i < function->get_parameters().size() - staticShape.size(); i++) {
                staticShape.push_back(staticShape.front());
            }
        }
    }
}
} // namespace subgraph
} // namespace test
} // namespace ov
