// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using ConvertToPluginSpecificNodeParams = std::tuple<ov::Shape,                      // non const input shape
                                                     ov::Shape,                      // const input shape
                                                     ov::element::Type,              // element type
                                                     ov::test::utils::EltwiseTypes,  // node type
                                                     size_t,                         // port for const input
                                                     size_t>;                        // expected number of constant node

class ConvertToPluginSpecificNode : public testing::WithParamInterface<ConvertToPluginSpecificNodeParams>,
                                    public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertToPluginSpecificNodeParams> obj) {
        ov::Shape nonConstShape, constShape;
        ov::element::Type prc;
        ov::test::utils::EltwiseTypes nodeType;
        size_t port, constNodeNum;
        std::tie(nonConstShape, constShape, prc, nodeType, port, constNodeNum) = obj.param;

        std::ostringstream result;
        result << "IS_NON_CONST=" << nonConstShape << "_";
        result << "IS_CONST=" << constShape << "_";
        result << "PRC=" << prc << "_";
        result << "NODE=" << nodeType << "_";
        result << "PORT=" << port << "_";
        result << "CONST_NUM=" << constNodeNum;

        return result.str();
    }

protected:
    size_t constNodeNum;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::Shape nonConstShape, constShape;
        ov::element::Type prc;
        ov::test::utils::EltwiseTypes nodeType;
        size_t port;

        std::tie(nonConstShape, constShape, prc, nodeType, port, constNodeNum) = this->GetParam();
        OPENVINO_ASSERT(shape_size(constShape) == 1);

        const auto param = std::make_shared<ov::op::v0::Parameter>(prc, ov::Shape(nonConstShape));
        const auto constNode = ov::test::utils::make_constant(prc, constShape, utils::InputGenerateData(1, 9e8, 1, 1));
        OutputVector inputs(2);
        inputs[port] = constNode;
        inputs[1 - port] = param;

        auto powerStatic = ov::test::utils::make_eltwise(inputs[0], inputs[1], nodeType);

        function = std::make_shared<ov::Model>(powerStatic, ParameterVector{param}, "ConvertToPluginSpecificNode");
    }
};

TEST_P(ConvertToPluginSpecificNode, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Const", constNodeNum);
}

namespace {

const std::vector<ov::Shape> nonConstIS = {{3, 4, 5, 6}};

const std::vector<ov::Shape> constIS = {
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
};

std::vector<ov::test::utils::EltwiseTypes> nodeTypes = {ov::test::utils::EltwiseTypes::ADD,
                                                        ov::test::utils::EltwiseTypes::SUBTRACT,
                                                        ov::test::utils::EltwiseTypes::MULTIPLY};

const std::vector<size_t> port = {0, 1};

const auto testParamsEltwise = ::testing::Combine(::testing::ValuesIn(nonConstIS),
                                                  ::testing::ValuesIn(constIS),
                                                  ::testing::Values(ov::element::f32),
                                                  ::testing::ValuesIn(nodeTypes),
                                                  ::testing::ValuesIn(port),
                                                  ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_CheckEltwise,
                         ConvertToPluginSpecificNode,
                         testParamsEltwise,
                         ConvertToPluginSpecificNode::getTestCaseName);

const auto testParamsPower = ::testing::Combine(::testing::ValuesIn(nonConstIS),
                                                ::testing::ValuesIn(constIS),
                                                ::testing::Values(ov::element::f32),
                                                ::testing::Values(ov::test::utils::EltwiseTypes::POWER),
                                                ::testing::Values(1),
                                                ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_CheckPower,
                         ConvertToPluginSpecificNode,
                         testParamsPower,
                         ConvertToPluginSpecificNode::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
