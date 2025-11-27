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
    std::cerr << "[DEBUG_CVS-172561] createGraph: type=" << type << ", shape=" << ov::test::utils::vec2str(input_shape) 
              << ", element_type=" << input_type << std::endl;
    switch (type) {
    case ConstantSubgraphType::SINGLE_COMPONENT: {
        std::cerr << "[DEBUG_CVS-172561] Creating SINGLE_COMPONENT graph" << std::endl;
        auto input = ov::test::utils::make_constant(input_type, input_shape);
        std::cerr << "[DEBUG_CVS-172561] Constant created, adding Result node" << std::endl;
        results.push_back(std::make_shared<ov::op::v0::Result>(input));
        break;
    }
    case ConstantSubgraphType::SEVERAL_COMPONENT: {
        std::cerr << "[DEBUG_CVS-172561] Creating SEVERAL_COMPONENT graph" << std::endl;
        auto input1 = ov::test::utils::make_constant(input_type, input_shape);
        std::cerr << "[DEBUG_CVS-172561] Constant 1 created, adding Result node" << std::endl;
        results.push_back(std::make_shared<ov::op::v0::Result>(input1));
        auto input2 = ov::test::utils::make_constant(input_type, input_shape);
        std::cerr << "[DEBUG_CVS-172561] Constant 2 created, adding Result node" << std::endl;
        results.push_back(std::make_shared<ov::op::v0::Result>(input2));
        break;
    }
    default: {
        throw std::runtime_error("Unsupported constant graph type");
    }
    }
    std::cerr << "[DEBUG_CVS-172561] Creating Model with " << results.size() << " results and " << params.size() << " params" << std::endl;
    function = std::make_shared<ov::Model>(results, params, "ConstResult");
    std::cerr << "[DEBUG_CVS-172561] Model created successfully" << std::endl;
    std::cerr.flush();
}

void ConstantResultSubgraphTest::SetUp() {
    const auto& [type, input_shape, input_type, _targetDevice] = this->GetParam();
    std::cerr << "[DEBUG_CVS-172561] ConstantResultSubgraphTest::SetUp() started" << std::endl;
    std::cerr << "[DEBUG_CVS-172561] SubgraphType: " << type << ", InputShape: " << ov::test::utils::vec2str(input_shape) 
              << ", InputType: " << input_type << ", TargetDevice: " << _targetDevice << std::endl;
    targetDevice = _targetDevice;

    std::cerr << "[DEBUG_CVS-172561] Before createGraph()" << std::endl;
    createGraph(type, input_shape, input_type);
    std::cerr << "[DEBUG_CVS-172561] After createGraph() - SUCCESS" << std::endl;
    std::cerr.flush();
}
}  // namespace test
}  // namespace ov
