// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/activation.hpp"

#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
using ov::test::utils::ActivationTypes;
void ActivationLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    ov::element::Type model_type;
    std::pair<std::vector<InputShape>, ov::Shape> input_shapes;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, model_type, input_shapes, targetDevice) = GetParam();

    bool inPrcSigned = function->get_parameters()[0]->get_element_type().is_signed();
    int32_t data_start_from;
    uint32_t data_range;
    int32_t resolution;

    switch (activationDecl.first) {
        case ActivationTypes::Log: {
            data_start_from = 1;
            data_range = 20;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Sqrt: {
            data_start_from = 0;
            data_range = 20;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Asin: {
            data_start_from = -1;
            data_range = 2;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Acos: {
            data_start_from = -1;
            data_range = 2;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Acosh: {
            data_start_from = 1;
            data_range = 200;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Atanh: {
            data_start_from = -1;
            data_range = 2;
            resolution = 32768;
            break;
        }
        case ActivationTypes::Ceiling: {
            data_start_from = -1000;
            data_range = 2000;
            resolution = 32768;
            break;
        }
        case ActivationTypes::RoundHalfToEven: {
            data_start_from = -10;
            data_range = 20;
            resolution = 4;
            break;
        }
        case ActivationTypes::RoundHalfAwayFromZero: {
            data_start_from = -10;
            data_range = 20;
            resolution = 4;
            break;
        }
        case ActivationTypes::Mish: {
            data_start_from = -20;
            data_range = 60;
            resolution = 32768;
            break;
        }
        case ActivationTypes::SoftPlus: {
            data_start_from = -100;
            data_range = 200;
            resolution = 32768;
            break;
        }
        case ActivationTypes::SoftSign: {
            data_start_from = -100;
            data_range = 200;
            resolution = 32768;
            break;
        }
        default: {
            data_start_from = -10;
            data_range = 20;
            resolution = 32768;
            break;
        }
    }
    if (!inPrcSigned) {
        data_range = 15;
        data_start_from = 0;
    }
    const auto& funcInputs = function->inputs();
    auto funcInput = funcInputs.begin();
    inputs.clear();
    Tensor data_tensor = ov::test::utils::create_and_fill_tensor_act_dft(funcInput->get_element_type(),
                                            targetInputStaticShapes[0],
                                            data_range,
                                            data_start_from,
                                            resolution, 1);
    inputs.insert({funcInput->get_node_shared_ptr(), data_tensor});
}

void ActivationParamLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    ov::element::Type model_type;
    std::pair<std::vector<InputShape>, ov::Shape> input_shapes;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, model_type, input_shapes, targetDevice) = GetParam();

    auto activationType = activationDecl.first;
    auto constants_value = activationDecl.second;

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        Tensor data_tensor;
        const std::string& name = funcInput.get_node()->get_friendly_name();
        if (0 == name.compare("negativeSlope")) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -0.01;
            in_data.range = 0;
            data_tensor = ov::test::utils::create_and_fill_tensor_act_dft(funcInput.get_element_type(),
                                            targetInputStaticShapes[i],
                                            in_data.range,
                                            in_data.start_from,
                                            in_data.resolution, 1);
        } else if (0 == name.compare("leakySlope")) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0.01;
            in_data.range = 0;
            data_tensor = ov::test::utils::create_and_fill_tensor_act_dft(funcInput.get_element_type(),
                                            targetInputStaticShapes[i],
                                            in_data.range,
                                            in_data.start_from,
                                            in_data.resolution, 1);
        } else if (0 == name.compare("alpha")) {
            data_tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], &constants_value[0]);
        } else if (0 == name.compare("beta") || 0 == name.compare("lambda")) {
            data_tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i], &constants_value[1]);
        } else {
            data_tensor = ov::test::utils::create_and_fill_tensor_act_dft(funcInput.get_element_type(), targetInputStaticShapes[i], 20, -10, 1);
        }
        inputs.insert({funcInput.get_node_shared_ptr(), data_tensor});
    }
}

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

    auto activation = ov::test::utils::make_activation(param, model_type, activationType, const_shape, constants_value);

    auto result = std::make_shared<ov::op::v0::Result>(activation);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "Activation");

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
    abs_threshold = 9e-4;
#else
    if (model_type == ov::element::f32) {
        abs_threshold = 1e-5;
    } else if (model_type == ov::element::f16) {
        abs_threshold = 5e-4;
    }
#endif
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

    auto activation = ov::test::utils::make_activation(params, model_type, activationType);
    auto result = std::make_shared<ov::op::v0::Result>(activation);
    function = std::make_shared<ov::Model>(result, params);
}
}  // namespace test
}  // namespace ov
