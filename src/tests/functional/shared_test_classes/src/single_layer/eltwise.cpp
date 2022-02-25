// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/eltwise.hpp"

#include "functional_test_utils/plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string EltwiseLayerTest::getTestCaseName(const testing::TestParamInfo<EltwiseTestParams>& obj) {
    std::vector<InputShape> shapes;
    ElementType netType, inType, outType;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseOpType;
    std::string targetName;
    ov::AnyMap additional_config;
    std::tie(shapes, eltwiseOpType, secondaryInputType, opType, netType, inType, outType, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=(";
    for (const auto& shape : shapes) {
        results << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    results << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
    }
    results << ")_eltwiseOpType=" << eltwiseOpType << "_";
    results << "secondaryInputType=" << secondaryInputType << "_";
    results << "opType=" << opType << "_";
    results << "NetType=" << netType << "_";
    results << "InType=" << inType << "_";
    results << "OutType=" << outType << "_";
    results << "trgDev=" << targetName;
    for (auto const& configItem : additional_config) {
        results << "_configItem=" << configItem.first << "_";
        configItem.second.print(results);
    }
    return results.str();
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
    if (!secondInputShape.is_static()) {
        return;
    }
    if (secondInputShape.get_shape() == ov::Shape{1}) {
        inputDynamicShapes[1] = secondInputShape;
        for (auto& staticShape : targetStaticShapes) {
            staticShape[1] = secondInputShape.get_shape();
        }
    }
}

void EltwiseLayerTest::SetUp() {
    // TODO: Remove it after fixing issue 69529
    // w/a for myriad (cann't store 2 caches simultaneously)
    PluginCache::get().reset();

    std::vector<InputShape> shapes;
    ElementType netType;
    ngraph::helpers::InputLayerType secondaryInputType;
    CommonTestUtils::OpType opType;
    ngraph::helpers::EltwiseTypes eltwiseType;
    Config additional_config;
    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, inType, outType[0], targetDevice, configuration) =
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
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        transformInputShapesAccordingEltwise(shape_input_secondary);
    }

    std::shared_ptr<ngraph::Node> secondaryInput;
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        secondaryInput = ngraph::builder::makeDynamicParams(netType, {shape_input_secondary}).front();
        parameters.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    } else {
        ov::Shape shape = inputDynamicShapes.back().get_max_shape();
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

} //  namespace subgraph
} //  namespace test
} //  namespace ov
