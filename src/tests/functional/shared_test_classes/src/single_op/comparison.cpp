// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/comparison.hpp"

#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
using ov::test::utils::ComparisonTypes;
using ov::test::utils::InputLayerType;

std::string ComparisonLayerTest::getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj) {
    std::vector<InputShape> shapes;
    ComparisonTypes comparison_op_type;
    InputLayerType second_input_type;
    ov::element::Type model_type;
    std::string device_name;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes,
             comparison_op_type,
             second_input_type,
             model_type,
             device_name,
             additional_config) = obj.param;

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
    result << "comparisonOpType=" << comparison_op_type << "_";
    result << "secondInputType=" << second_input_type << "_";
    result << "in_type=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << device_name;
    return result.str();
}

void ComparisonLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    InputLayerType second_input_type;
    std::map<std::string, std::string> additional_config;
    ov::element::Type model_type;
    ov::test::utils::ComparisonTypes comparison_op_type;
    std::tie(shapes,
             comparison_op_type,
             second_input_type,
             model_type,
             targetDevice,
             additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    init_input_shapes(shapes);

    ov::ParameterVector inputs {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};

    std::shared_ptr<ov::Node> second_input;
    if (second_input_type == InputLayerType::PARAMETER) {
        second_input = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
        inputs.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(second_input));
    } else {
        ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_type, targetStaticShapes.front()[1]);
        second_input = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto comparisonNode = ov::test::utils::make_comparison(inputs[0], second_input, comparison_op_type);
    function = std::make_shared<ov::Model>(comparisonNode, inputs, "Comparison");
}
} // namespace test
} // namespace ov
