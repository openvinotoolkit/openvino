// Copyright (C) 2018-2025 Intel Corporation
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
    ConstantSubgraphType type;
    ov::Shape input_shape;
    ov::element::Type input_type;
    std::string target_device;

    std::tie(type, input_shape, input_type, target_device) = obj.param;
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
    ConstantSubgraphType type;
    ov::Shape input_shape;
    ov::element::Type input_type;
    std::tie(type, input_shape, input_type, targetDevice) = this->GetParam();

    createGraph(type, input_shape, input_type);
}
}  // namespace test
}  // namespace ov
