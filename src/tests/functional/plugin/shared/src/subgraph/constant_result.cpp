// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

std::ostream& operator<<(std::ostream& os, ConstantSubgraphType type) {
    switch (type) {
    case ConstantSubgraphType::SINGLE_COMPONENT:
        os << "SINGLE_COMPONENT";
        break;
    case ConstantSubgraphType::SEVERAL_COMPONENT:
        os << "SEVERAL_COMPONENT";
        break;
    default:
        os << "UNSUPPORTED_CONST_SUBGRAPH_TYPE";
    }
    return os;
}

std::string ConstantResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<constResultParams>& obj) {
    const auto& [type, input_shape, input_type, target_device] = obj.param;
    std::ostringstream result;
    result << "SubgraphType=" << type << "_";
    result << "IS=" << input_shape << "_";
    result << "IT=" << input_type << "_";
    result << "Device=" << target_device;
    return result.str();
}

void ConstantResultSubgraphTest::createGraph(const ConstantSubgraphType& type,
                                             const ov::Shape& input_shape,
                                             const ov::element::Type& input_type) {
    ParameterVector params;
    ResultVector results;
    switch (type) {
    case ConstantSubgraphType::SINGLE_COMPONENT: {
        auto input = ov::test::utils::make_constant(input_type, input_shape);
        results.push_back(std::make_shared<ov::op::v0::Result>(input));
        break;
    }
    case ConstantSubgraphType::SEVERAL_COMPONENT: {
        auto input1 = ov::test::utils::make_constant(input_type, input_shape);
        results.push_back(std::make_shared<ov::op::v0::Result>(input1));
        auto input2 = ov::test::utils::make_constant(input_type, input_shape);
        results.push_back(std::make_shared<ov::op::v0::Result>(input2));
        break;
    }
    default: {
        throw std::runtime_error("Unsupported constant graph type");
    }
    }
    function = std::make_shared<ov::Model>(results, params, "ConstResult");
}

void ConstantResultSubgraphTest::SetUp() {
    const auto& [type, input_shape, input_type, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    createGraph(type, input_shape, input_type);
}

void ConstantResultSubgraphTest::run() {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);
    inferRequest.infer();

    const auto& [type, input_shape, input_type, _] = this->GetParam();
    if (input_type == ov::element::i16 || input_type == ov::element::u16) {
        auto outputs = function->get_results();
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto result_tensor = inferRequest.get_tensor(outputs[i]);
            ASSERT_TRUE(result_tensor);

            auto constant_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                outputs[i]->get_input_node_shared_ptr(0));
            ASSERT_TRUE(constant_node) << "Failed to get constant node for output " << i;

            size_t num_elements = result_tensor.get_size();
            ASSERT_EQ(result_tensor.get_element_type(), input_type)
                << "Output type mismatch for " << input_type;

            if (input_type == ov::element::i16) {
                auto expected_data = constant_node->get_data_ptr<int16_t>();
                auto actual_data = result_tensor.data<int16_t>();
                for (size_t j = 0; j < num_elements; ++j) {
                    EXPECT_EQ(actual_data[j], expected_data[j])
                        << "Mismatch at element " << j << "/" << num_elements
                        << ": expected " << expected_data[j]
                        << ", got " << actual_data[j];
                }
            } else {
                auto expected_data = constant_node->get_data_ptr<uint16_t>();
                auto actual_data = result_tensor.data<uint16_t>();
                for (size_t j = 0; j < num_elements; ++j) {
                    EXPECT_EQ(actual_data[j], expected_data[j])
                        << "Mismatch at element " << j << "/" << num_elements
                        << ": expected " << expected_data[j]
                        << ", got " << actual_data[j];
                }
            }
        }
    }
}
}  // namespace test
}  // namespace ov
