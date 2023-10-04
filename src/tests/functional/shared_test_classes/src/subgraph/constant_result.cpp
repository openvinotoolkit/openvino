// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

#include "ngraph_functions/builders.hpp"
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
        auto input = ngraph::builder::makeConstant<float>(input_type, input_shape, {}, true);
        results.push_back(std::make_shared<ov::op::v0::Result>(input));
        break;
    }
    case ConstantSubgraphType::SEVERAL_COMPONENT: {
        auto input1 = ngraph::builder::makeConstant<float>(input_type, input_shape, {}, true);
        results.push_back(std::make_shared<ov::op::v0::Result>(input1));
        auto input2 = ngraph::builder::makeConstant<float>(input_type, input_shape, {}, true);
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

namespace SubgraphTestsDefinitions {

std::string ConstantResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<constResultParams>& obj) {
    ConstantSubgraphType type;
    InferenceEngine::SizeVector IS;
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;

    std::tie(type, IS, inputPrecision, targetDevice) = obj.param;
    std::ostringstream result;
    result << "SubgraphType=" << type << "_";
    result << "IS=" << ov::test::utils::vec2str(IS) << "_";
    result << "inPrc=" << inputPrecision << "_";
    result << "Device=" << targetDevice;
    return result.str();
}

void ConstantResultSubgraphTest::createGraph(const ConstantSubgraphType& type,
                                             const InferenceEngine::SizeVector& inputShape,
                                             const InferenceEngine::Precision& inputPrecision) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);

    ov::ParameterVector params;
    ov::ResultVector results;
    switch (type) {
    case ConstantSubgraphType::SINGLE_COMPONENT: {
        auto input = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {}, true);
        results.push_back(std::make_shared<ov::op::v0::Result>(input));
        break;
    }
    case ConstantSubgraphType::SEVERAL_COMPONENT: {
        auto input1 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {}, true);
        results.push_back(std::make_shared<ov::op::v0::Result>(input1));
        auto input2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {}, true);
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
    InferenceEngine::SizeVector IS;
    InferenceEngine::Precision inputPrecision;
    std::tie(type, IS, inputPrecision, targetDevice) = this->GetParam();

    createGraph(type, IS, inputPrecision);
}

}  // namespace SubgraphTestsDefinitions
