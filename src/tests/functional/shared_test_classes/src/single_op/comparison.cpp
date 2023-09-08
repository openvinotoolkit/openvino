// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/comparison.hpp"

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
using ngraph::helpers::ComparisonTypes;
using ngraph::helpers::InputLayerType;

std::string ComparisonLayerTest::getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj) {
    std::vector<InputShape> shapes;
    ComparisonTypes comparison_op_type;
    InputLayerType second_input_type;
    ov::element::Type in_type;
    std::string device_name;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes,
             comparison_op_type,
             second_input_type,
             in_type,
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
    result << "in_type=" << in_type.get_type_name() << "_";
    result << "targetDevice=" << device_name;
    return result.str();
}

void ComparisonLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    InputLayerType second_input_type;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes,
             comparison_op_type,
             second_input_type,
             inType,
             targetDevice,
             additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    init_input_shapes(shapes);
    outType = inType;

    ov::ParameterVector inputs {std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0])};

    std::shared_ptr<ov::Node> second_input;
    if (second_input_type == InputLayerType::PARAMETER) {
        second_input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
        inputs.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(second_input));
    } else {
        ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(inType, targetStaticShapes.front()[1]);
        second_input = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto comparisonNode = ngraph::builder::makeComparison(inputs[0], second_input, comparison_op_type);
    function = std::make_shared<ov::Model>(comparisonNode, inputs, "Comparison");
}

void ComparisonLayerTest::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
    if (comparison_op_type == ComparisonTypes::IS_FINITE || comparison_op_type == ComparisonTypes::IS_NAN) {
        inputs.clear();
        auto params = function->get_parameters();
        OPENVINO_ASSERT(target_input_static_shapes.size() >= params.size());
        for (int i = 0; i < params.size(); i++) {
            ov::Tensor tensor(params[i]->get_element_type(), target_input_static_shapes[i]);
            auto data_ptr = static_cast<float*>(tensor.data());
            auto data_ptr_int = static_cast<int*>(tensor.data());
            auto range = tensor.get_size();
            auto start = -static_cast<float>(range) / 2.f;
            testing::internal::Random random(1);
            for (size_t i = 0; i < range; i++) {
                if (i % 7 == 0) {
                    data_ptr[i] = std::numeric_limits<float>::infinity();
                } else if (i % 7 == 1) {
                    data_ptr[i] = -std::numeric_limits<float>::infinity();
                } else if (i % 7 == 2) {
                    data_ptr_int[i] = 0x7F800000 + random.Generate(range);
                } else if (i % 7 == 3) {
                    data_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                } else if (i % 7 == 5) {
                    data_ptr[i] = -std::numeric_limits<double>::quiet_NaN();
                } else {
                    data_ptr[i] = start + static_cast<float>(random.Generate(range));
                }
            }
            inputs.insert({params[i], tensor});
        }
    } else {
        SubgraphBaseTest::generate_inputs(target_input_static_shapes);
    }
}

} // namespace test
} // namespace ov
