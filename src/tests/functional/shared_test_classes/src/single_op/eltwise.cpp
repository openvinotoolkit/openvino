// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"

#include "shared_test_classes/single_op/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

namespace ov {
namespace test {
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;
using ov::test::utils::EltwiseTypes;

std::string EltwiseLayerTest::getTestCaseName(const testing::TestParamInfo<EltwiseTestParams>& obj) {
    std::vector<InputShape> shapes;
    ElementType model_type, in_type, out_type;
    InputLayerType secondary_input_type;
    OpType op_type;
    EltwiseTypes eltwise_op_type;
    std::string device_name;
    ov::AnyMap additional_config;
    std::tie(shapes, eltwise_op_type, secondary_input_type, op_type, model_type, in_type, out_type, device_name, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=(";
    for (const auto& shape : shapes) {
        results << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    results << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
    }
    results << ")_eltwise_op_type=" << eltwise_op_type << "_";
    results << "secondary_input_type=" << secondary_input_type << "_";
    results << "opType=" << op_type << "_";
    results << "model_type=" << model_type << "_";
    results << "InType=" << in_type << "_";
    results << "OutType=" << out_type << "_";
    results << "trgDev=" << device_name;
    for (auto const& config_item : additional_config) {
        results << "_config_item=" << config_item.first << "=";
        config_item.second.print(results);
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
    std::vector<InputShape> shapes;
    ElementType model_type;
    InputLayerType secondary_input_type;
    OpType op_type;
    EltwiseTypes eltwise_type;
    Config additional_config;
    std::tie(shapes, eltwise_type, secondary_input_type, op_type, model_type, inType, outType, targetDevice, configuration) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector parameters{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    ov::PartialShape shape_input_secondary;
    switch (op_type) {
        case OpType::SCALAR: {
            shape_input_secondary = {1};
            break;
        }
        case OpType::VECTOR:
            shape_input_secondary = inputDynamicShapes.back();
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }
    // To propagate shape_input_secondary just in static case because all shapes are defined in dynamic scenarion
    if (secondary_input_type == InputLayerType::PARAMETER) {
        transformInputShapesAccordingEltwise(shape_input_secondary);
    }

    std::shared_ptr<ov::Node> secondary_input;
    if (secondary_input_type == InputLayerType::PARAMETER) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape_input_secondary);
        secondary_input = param;
        parameters.push_back(param);
    } else {
        ov::Shape shape = inputDynamicShapes.back().get_max_shape();
        ov::test::utils::InputGenerateData in_data;
        switch (eltwise_type) {
            case EltwiseTypes::DIVIDE:
            case EltwiseTypes::MOD:
            case EltwiseTypes::FLOOR_MOD: {
                in_data.start_from = 2;
                in_data.range = 8;
                auto tensor = ov::test::utils::create_and_fill_tensor(model_type, shape, in_data);
                secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
                break;
            }
            case EltwiseTypes::POWER: {
                in_data.start_from = 1;
                in_data.range = 2;
                auto tensor = ov::test::utils::create_and_fill_tensor(model_type, shape, in_data);
                secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
                break;
            }
            case EltwiseTypes::LEFT_SHIFT:
            case EltwiseTypes::RIGHT_SHIFT: {
                in_data.start_from = 0;
                in_data.range = 4;
                auto tensor = ov::test::utils::create_and_fill_tensor(model_type, shape, in_data);
                secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
                break;
            }
            default: {
                in_data.start_from = 1;
                in_data.range = 9;
                auto tensor = ov::test::utils::create_and_fill_tensor(model_type, shape, in_data);
                secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
            }
        }
    }

    parameters[0]->set_friendly_name("param0");
    secondary_input->set_friendly_name("param1");

    auto eltwise = ov::test::utils::make_eltwise(parameters[0], secondary_input, eltwise_type);
    function = std::make_shared<ov::Model>(eltwise, parameters, "Eltwise");
}
} //  namespace test
} //  namespace ov
