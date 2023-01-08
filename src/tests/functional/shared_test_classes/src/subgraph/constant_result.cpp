// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace SubgraphTestsDefinitions {

std::ostream& operator<<(std::ostream &os, ConstantSubgraphType type) {
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
    SizeVector IS;
    Precision inputPrecision;
    std::string targetDevice;

    std::tie(type, IS, inputPrecision, targetDevice) = obj.param;
    std::ostringstream result;
    result << "SubgraphType=" << type << "_";
    result << "IS=" << CommonTestUtils::vec2str(IS) << "_";
    result << "inPrc=" << inputPrecision << "_";
    result << "Device=" << targetDevice;
    return result.str();
}

void ConstantResultSubgraphTest::createGraph(const ConstantSubgraphType& type, const SizeVector &inputShape, const Precision &inputPrecision) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);

    ParameterVector params;
    ResultVector results;
    switch (type) {
        case ConstantSubgraphType::SINGLE_COMPONENT: {
            auto input = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input));
            break;
        }
        case ConstantSubgraphType::SEVERAL_COMPONENT: {
            auto input1 = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input1));
            auto input2 = builder::makeConstant<float>(ngPrc, inputShape, {}, true);
            results.push_back(std::make_shared<opset3::Result>(input2));
            break;
        }
        default: {
            throw std::runtime_error("Unsupported constant graph type");
        }
    }
    function = std::make_shared<Function>(results, params, "ConstResult");
}

void ConstantResultSubgraphTest::SetUp() {
    ConstantSubgraphType type;
    SizeVector IS;
    Precision inputPrecision;
    std::tie(type, IS, inputPrecision, targetDevice) = this->GetParam();

    createGraph(type, IS, inputPrecision);
}

}  // namespace SubgraphTestsDefinitions
