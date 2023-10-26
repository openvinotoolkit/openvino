// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/activation.hpp"

#include "ov_models/builders.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
using ov::test::utils::ActivationTypes;

std::string ActivationLayerTest::getTestCaseName(const testing::TestParamInfo<activationParams> &obj) {
    ov::element::Type model_type;
    std::pair<std::vector<InputShape>, ov::Shape> input_shapes;
    std::string target_device;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, model_type, input_shapes, target_device) = obj.param;

    auto shapes = input_shapes.first;
    auto const_shape = input_shapes.second;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "TS=" << ov::test::utils::vec2str(const_shape) << separator;
    result << activationNames[activationDecl.first] << separator;
    result << "constants_value=" << ov::test::utils::vec2str(activationDecl.second) << separator;
    result << "netPRC=" << model_type.get_type_name() << separator;
    result << "trgDev=" << target_device;
    return result.str();
}

void ActivationLayerTest::SetUp() {
    ov::element::Type model_type;
    std::pair<std::vector<InputShape>, ov::Shape> input_shapes;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, model_type, input_shapes, targetDevice) = GetParam();
    init_input_shapes(input_shapes.first);
    auto const_shape = input_shapes.second;

    auto activationType = activationDecl.first;
    auto constants_value = activationDecl.second;

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    param->set_friendly_name("Input");

    if (activationType == ActivationTypes::PReLu && constants_value.empty()) {
        auto elemnts_count = ov::shape_size(const_shape);
        constants_value.resize(elemnts_count);
        std::iota(constants_value.begin(), constants_value.end(), -10);
    }

    auto activation = ngraph::builder::makeActivation(param, model_type, activationType, const_shape, constants_value);

    auto result = std::make_shared<ov::op::v0::Result>(activation);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "Activation");
}

void ActivationParamLayerTest::SetUp() {
    ov::element::Type model_type;
    std::pair<std::vector<InputShape>, ov::Shape> input_shapes;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, model_type, input_shapes, targetDevice) = GetParam();
    auto shapes = input_shapes.first;
    auto const_shape = input_shapes.second;

    auto activationType = activationDecl.first;
    auto constants_value = activationDecl.second;

    switch (activationType) {
        case ActivationTypes::PReLu:
        case ActivationTypes::LeakyRelu: {
            shapes.push_back(ov::test::static_shapes_to_test_representation({const_shape}).front());
            break;
        }
        case ActivationTypes::HardSigmoid:
        case ActivationTypes::Selu: {
            shapes.push_back(ov::test::static_shapes_to_test_representation({const_shape}).front());
            shapes.push_back(ov::test::static_shapes_to_test_representation({const_shape}).front());
            break;
        }
        default:
            OPENVINO_THROW("Unsupported activation type for Params test type");
    }

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (const auto& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    switch (activationType) {
        case ActivationTypes::PReLu: {
            params[1]->set_friendly_name("negativeSlope");
            break;
        }
        case ActivationTypes::LeakyRelu: {
            params[1]->set_friendly_name("leakySlope");
            break;
        }
        case ActivationTypes::HardSigmoid: {
            params[1]->set_friendly_name("alpha");
            params[2]->set_friendly_name("beta");
            break;
        }
        case ActivationTypes::Selu: {
            params[1]->set_friendly_name("alpha");
            params[2]->set_friendly_name("lambda");
            break;
        }
        default:
            OPENVINO_THROW("Unsupported activation type for Params test type");
    }

    params[0]->set_friendly_name("Input");

    auto activation = ngraph::builder::makeActivation(params, model_type, activationType);
    auto result = std::make_shared<ov::op::v0::Result>(activation);
    function = std::make_shared<ov::Model>(result, params);
}
}  // namespace test
}  // namespace ov
