// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/logical.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/logical.hpp"

namespace ov {
namespace test {
std::string LogicalLayerTest::getTestCaseName(const testing::TestParamInfo<LogicalTestParams>& obj) {
    std::vector<InputShape> shapes;
    ov::test::utils::LogicalTypes comparisonOpType;
    ov::test::utils::InputLayerType second_input_type;
    ov::element::Type model_type;
    std::string device_name;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, comparisonOpType, second_input_type, model_type, device_name, additional_config) = obj.param;

    std::ostringstream result;
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
    result << "comparisonOpType=" << comparisonOpType << "_";
    result << "second_input_type=" << second_input_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void LogicalLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::test::utils::LogicalTypes logical_op_type;
    ov::test::utils::InputLayerType second_input_type;
    ov::element::Type model_type;
    std::map<std::string, std::string> additional_config;

    std::tie(shapes, logical_op_type, second_input_type, model_type, targetDevice, additional_config) = this->GetParam();
    init_input_shapes(shapes);

    configuration.insert(additional_config.begin(), additional_config.end());

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};

    std::shared_ptr<ov::Node> logical_node;
    if (ov::test::utils::LogicalTypes::LOGICAL_NOT != logical_op_type) {
        std::shared_ptr<ov::Node> secondInput;
        if (ov::test::utils::InputLayerType::CONSTANT == second_input_type) {
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, targetStaticShapes[0][1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        } else {
            auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
            secondInput = param;
            params.push_back(param);
        }
        logical_node = ov::test::utils::make_logical(params[0], secondInput, logical_op_type);
    } else {
        logical_node = std::make_shared<ov::op::v1::LogicalNot>(params[0]);
    }

    function = std::make_shared<ov::Model>(logical_node, params, "Logical");
}
} // namespace test
} // namespace ov
