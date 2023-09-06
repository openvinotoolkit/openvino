// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

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
    result << "IS=" << ov::test::utils::vec2str(IS) << "_";
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


namespace ov {
namespace test {

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

std::string ConstantResultSubgraphTestNew::getTestCaseName(const testing::TestParamInfo<constResultParams>& obj) {
    ConstantSubgraphType type;
    std::vector<InputShape> shapes;
    ov::element::Type input_type;
    std::string targetDevice;

    std::tie(type, shapes, input_type, targetDevice) = obj.param;
    std::ostringstream result;
    result << "SubgraphType=" << type << "_";
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }    result << "IT=" << input_type.get_type_name() << "_";
    result << "Device=" << targetDevice;
    return result.str();
}

void ConstantResultSubgraphTestNew::createGraph(const ConstantSubgraphType type) {
    ParameterVector params;
    ResultVector results;
    switch (type) {
        case ConstantSubgraphType::SINGLE_COMPONENT: {
            auto tensor = ov::test::utils::create_and_fill_tensor(inType, targetStaticShapes.front()[0]);
            auto input = std::make_shared<ov::op::v0::Constant>(tensor);
            results.push_back(std::make_shared<opset3::Result>(input));
            break;
        }
        case ConstantSubgraphType::SEVERAL_COMPONENT: {
            auto tensor1 = ov::test::utils::create_and_fill_tensor(inType, targetStaticShapes.front()[0]);
            auto input1 = std::make_shared<ov::op::v0::Constant>(tensor1);
            results.push_back(std::make_shared<opset3::Result>(input1));
            auto tensor2 = ov::test::utils::create_and_fill_tensor(inType, targetStaticShapes.front()[0]);
            auto input2 = std::make_shared<ov::op::v0::Constant>(tensor2);
            results.push_back(std::make_shared<opset3::Result>(input2));
            break;
        }
        default: {
            throw std::runtime_error("Unsupported constant graph type");
        }
    }
    function = std::make_shared<ov::Model>(results, params, "ConstResult");
}

void ConstantResultSubgraphTestNew::SetUp() {
    ConstantSubgraphType type;
    std::vector<InputShape> shapes;
    std::tie(type, shapes, inType, targetDevice) = this->GetParam();
    init_input_shapes(shapes);
    outType = inType;

    createGraph(type);
}
} //  namespace test
} //  namespace ov