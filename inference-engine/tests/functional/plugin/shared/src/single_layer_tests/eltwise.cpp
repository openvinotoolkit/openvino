// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"
#include "common_test_utils/xml_net_builder/xml_filler.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ie_core.hpp"
#include "single_layer_tests/eltwise.hpp"

using namespace EltwiseTestNamespace;

std::string EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<eltwiseLayerTestParamsSet> obj) {
    EltwiseOpType op;
    ParameterInputIdx primary_input_idx;
    InputLayerType secondary_input_type;
    InferenceEngine::Precision prec;
    InferenceEngine::SizeVector vec;
    LayerTestsUtils::TargetDevice dev;
    std::map<std::string, std::string> additional_config;
    std::tie(op, primary_input_idx, secondary_input_type, prec, vec, dev, additional_config) = obj.param;

    std::ostringstream result;
    result << "operation=" << EltwiseOpType_to_string(op) << "_";
    result << "netPRC=" << prec.name() << "_";
    result << "primaryInputIdx=" << primary_input_idx << "_";
    result << "secondaryInputType=" << InputLayerType_to_string(secondary_input_type) << "_";
    result << "inputShapes=" << CommonTestUtils::vec2str(vec) << "_";
    result << "targetDevice=" << dev;
    return result.str();
}

void EltwiseLayerTest::SetUp() {
    EltwiseOpType op;
    ParameterInputIdx primary_input_idx;
    InputLayerType secondary_input_type;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::ParameterVector parameter_inputs;
    std::map<std::string, std::string> additional_config;
    std::tie(op, primary_input_idx, secondary_input_type, netPrecision, inputShape, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::shared_ptr<ngraph::Node> input0_node;
    std::shared_ptr<ngraph::Node> input1_node;
    auto primary_input = ngraph::builder::makeParams(ngPrc, { inputShape })[0];

    switch (secondary_input_type) {
    case InputLayerType::CONSTANT:
    {
        auto shape_total = 1;
        for (auto dim : inputShape) {
            shape_total *= dim;
        }

        const float min = -10;
        const float max = 10;
        const float range = max - min;
        const float step = range / shape_total;

        std::vector<float> const_vec(shape_total);
        for (int i = 0; i < shape_total; i++) {
            const_vec[i] = min + step * i;
        }

        auto const_vals = ngraph::builder::makeConstant(ngPrc, inputShape, const_vec);
        parameter_inputs.push_back(primary_input);

        if (primary_input_idx == 0) {
            input0_node = primary_input;
            input1_node = const_vals;
        } else {
            input0_node = const_vals;
            input1_node = primary_input;
        }
        break;
    }
    case InputLayerType::PARAMETER:
    {
        auto secondary_input = ngraph::builder::makeParams(ngPrc, { inputShape })[0];
        if (primary_input_idx == 0) {
            parameter_inputs.push_back(primary_input);
            parameter_inputs.push_back(secondary_input);
            input0_node = primary_input;
            input1_node = secondary_input;
        } else {
            parameter_inputs.push_back(secondary_input);
            parameter_inputs.push_back(primary_input);
            input0_node = secondary_input;
            input1_node = primary_input;
        }
        break;
    }
    default:
        ASSERT_EQ("unknown input type", "");
        break;
    }

    std::shared_ptr<ngraph::op::util::BinaryElementwiseArithmetic> ngraph_op = nullptr;
    switch (op) {
    case EltwiseOpType::ADD:
        ngraph_op = std::make_shared<ngraph::op::Add>(input0_node, input1_node);
        break;
    case EltwiseOpType::MULTIPLY:
        ngraph_op = std::make_shared<ngraph::op::Multiply>(input0_node, input1_node);
        break;
    case EltwiseOpType::SUBSTRACT:
        ngraph_op = std::make_shared<ngraph::op::Subtract>(input0_node, input1_node);
        break;
    default:
        ASSERT_EQ(std::string("Unknown Eltwise operation type: ") + EltwiseOpType_to_string(op), "");
        break;
    }
    function = std::make_shared<ngraph::Function>(ngraph_op, parameter_inputs, "Eltwise_op");
}

const char* EltwiseTestNamespace::InputLayerType_to_string(InputLayerType lt) {
    switch (lt) {
    case InputLayerType::CONSTANT:
        return "CONSTANT";
    case InputLayerType::PARAMETER:
        return "PARAMETER";
    default:
        return "NOT_SUPPORTED_INPUT_LAYER_TYPE";
    }
}

const char* EltwiseTestNamespace::EltwiseOpType_to_string(EltwiseOpType eOp) {
    switch (eOp) {
    case EltwiseOpType::ADD:
        return "Sum";
    case EltwiseOpType::MULTIPLY:
        return "Prod";
    case EltwiseOpType::SUBSTRACT:
        return "Sub";
    default:
        return "NOT_SUPPORTED_ELTWISE_OPERATION";
    }
}

TEST_P(EltwiseLayerTest, basic) {
    Run();
}
