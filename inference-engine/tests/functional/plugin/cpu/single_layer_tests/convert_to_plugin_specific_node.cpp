// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using ConvertToPluginSpecificNodeParams = std::tuple<SizeVector,            // non const input shape
                                                     SizeVector,            // const input shape
                                                     Precision,             // precision
                                                     helpers::EltwiseTypes, // node type
                                                     size_t,                // port for const input
                                                     size_t>;               // expected number of constant node

class ConvertToPluginSpecificNode : public testing::WithParamInterface<ConvertToPluginSpecificNodeParams>,
                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertToPluginSpecificNodeParams> obj) {
        SizeVector nonConstShape, constShape;
        Precision prc;
        helpers::EltwiseTypes nodeType;
        size_t port, constNodeNum;
        std::tie(nonConstShape, constShape, prc, nodeType, port, constNodeNum) = obj.param;

        std::ostringstream result;
        result << "IS_NON_CONST=" << CommonTestUtils::vec2str(nonConstShape) << "_";
        result << "IS_CONST=" << CommonTestUtils::vec2str(constShape) << "_";
        result << "PRC=" << prc << "_";
        result << "NODE=" << nodeType << "_";
        result << "PORT=" << port << "_";
        result << "CONST_NUM=" << constNodeNum;

        return result.str();
    }

protected:
    size_t constNodeNum;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        SizeVector nonConstShape, constShape;
        Precision prc;
        helpers::EltwiseTypes nodeType;
        size_t port;

        std::tie(nonConstShape, constShape, prc, nodeType, port, constNodeNum) = this->GetParam();
        IE_ASSERT(shape_size(constShape) == 1);

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
        const auto param = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(nonConstShape));
        const auto constNode = builder::makeConstant(ngPrc, ngraph::Shape(constShape), std::vector<float>{}, true);
        OutputVector inputs(2);
        inputs[port] = constNode;
        inputs[1 - port] = param;

        auto powerStatic = ngraph::builder::makeEltwise(inputs[0], inputs[1], nodeType);

        function = std::make_shared<ngraph::Function>(powerStatic, ParameterVector{param}, "ConvertToPluginSpecificNode");
    }
};

TEST_P(ConvertToPluginSpecificNode, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckNodeOfTypeCount(executableNetwork, "Const", constNodeNum);
}

namespace {

const std::vector<std::vector<size_t>> nonConstIS = {
    {3, 4, 5, 6}
};

const std::vector<std::vector<size_t>> constIS = {
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
};

std::vector<ngraph::helpers::EltwiseTypes> nodeTypes = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::MULTIPLY
};

const std::vector<size_t> port = {
    0, 1
};

const auto testParamsEltwise = ::testing::Combine(::testing::ValuesIn(nonConstIS),
                                                  ::testing::ValuesIn(constIS),
                                                  ::testing::Values(Precision::FP32),
                                                  ::testing::ValuesIn(nodeTypes),
                                                  ::testing::ValuesIn(port),
                                                  ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_CheckEltwise, ConvertToPluginSpecificNode, testParamsEltwise, ConvertToPluginSpecificNode::getTestCaseName);

const auto testParamsPower = ::testing::Combine(::testing::ValuesIn(nonConstIS),
                                                ::testing::ValuesIn(constIS),
                                                ::testing::Values(Precision::FP32),
                                                ::testing::Values(ngraph::helpers::EltwiseTypes::POWER),
                                                ::testing::Values(1),
                                                ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_CheckPower, ConvertToPluginSpecificNode, testParamsPower, ConvertToPluginSpecificNode::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
