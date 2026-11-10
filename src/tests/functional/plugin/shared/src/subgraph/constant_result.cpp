// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
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
    auto outputs = function->get_results();
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto result_tensor = inferRequest.get_tensor(outputs[i]);
        ASSERT_TRUE(result_tensor);

        auto constant_node = ov::as_type_ptr<ov::op::v0::Constant>(outputs[i]->get_input_node_shared_ptr(0));
        ASSERT_TRUE(constant_node) << "Failed to get constant node for output " << i;

        ASSERT_EQ(result_tensor.get_element_type(), input_type) << "Output type mismatch for " << input_type;
        ASSERT_EQ(result_tensor.get_size(), ov::shape_size(constant_node->get_shape())) << "Output size mismatch for output " << i;

        ov::Tensor expected(constant_node->get_element_type(),
                            constant_node->get_shape(),
                            const_cast<void*>(constant_node->get_data_ptr()));
        ov::test::utils::compare(expected, result_tensor, ov::element::f32);
    }
}
}  // namespace test
}  // namespace ov
