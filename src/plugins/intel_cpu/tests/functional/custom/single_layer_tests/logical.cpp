// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/logical.hpp"

#include "common_test_utils/node_builders/logical.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<ov::test::LogicalTestParams, CPUSpecificParams> LogicalLayerCPUTestParamSet;

class LogicalLayerCPUTest : public testing::WithParamInterface<LogicalLayerCPUTestParamSet>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LogicalLayerCPUTestParamSet> obj) {
        ov::test::LogicalTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << ov::test::LogicalLayerTest::getTestCaseName(
            testing::TestParamInfo<ov::test::LogicalTestParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        ov::test::LogicalTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::vector<ov::test::InputShape> inputShapes;
        ov::test::utils::LogicalTypes logicalOpType;
        ov::test::utils::InputLayerType secondInputType;
        ov::element::Type netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(inputShapes, logicalOpType, secondInputType, netPrecision, targetDevice, additional_config) =
            basicParamsSet;
        init_input_shapes(inputShapes);

        selectedType = getPrimitiveType() + "_" + ov::element::Type(inType).get_type_name();

        auto ngInputsPrc = ov::element::boolean;  // Because ngraph supports only boolean input for logical ops
        configuration.insert(additional_config.begin(), additional_config.end());

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, inputDynamicShapes[0])};
        std::shared_ptr<ov::Node> logicalNode;
        if (logicalOpType != ov::test::utils::LogicalTypes::LOGICAL_NOT) {
            std::shared_ptr<ov::Node> secondInput;
            if (secondInputType == ov::test::utils::InputLayerType::PARAMETER) {
                auto param = std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, inputDynamicShapes[1]);
                secondInput = param;
                inputs.push_back(param);
            } else {
                auto tensor = ov::test::utils::create_and_fill_tensor(ngInputsPrc, targetStaticShapes[0][1]);
                secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
            }
            logicalNode = ov::test::utils::make_logical(inputs[0], secondInput, logicalOpType);
        } else {
            logicalNode = ov::test::utils::make_logical(inputs[0], ov::Output<ov::Node>(), logicalOpType);
        }

        logicalNode->get_rt_info() = getCPUInfo();

        function = std::make_shared<ov::Model>(logicalNode, inputs, "Logical");
    }
};

TEST_P(LogicalLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

namespace {

std::map<ov::Shape, std::vector<ov::Shape>> inputShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
    {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
    {{1, 3, 20}, {{20}, {2, 1, 1}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
    {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> inputShapesNot = {
    {{1}, {}},
    {{5}, {}},
    {{2, 200}, {}},
    {{1, 3, 20}, {}},
    {{2, 17, 3, 4}, {}},
    {{2, 1, 1, 3, 1}, {}},
};

std::vector<ov::test::utils::LogicalTypes> logicalOpTypes = {
    ov::test::utils::LogicalTypes::LOGICAL_AND,
    ov::test::utils::LogicalTypes::LOGICAL_OR,
    ov::test::utils::LogicalTypes::LOGICAL_XOR,
};

std::vector<ov::test::utils::InputLayerType> secondInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config;

std::vector<std::vector<ov::Shape>> combine_shapes(
    const std::map<ov::Shape, std::vector<ov::Shape>>& input_shapes_static) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& input_shape : input_shapes_static) {
        for (auto& item : input_shape.second) {
            result.push_back({input_shape.first, item});
        }

        if (input_shape.second.empty()) {
            result.push_back({input_shape.first, {}});
        }
    }
    return result;
}

const auto LogicalTestParams = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(combine_shapes(inputShapes))),
                       ::testing::ValuesIn(logicalOpTypes),
                       ::testing::ValuesIn(secondInputTypes),
                       ::testing::Values(ov::element::bf16),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(additional_config)),
    ::testing::Values(emptyCPUSpec));

const auto LogicalTestParamsNot = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(combine_shapes(inputShapesNot))),
        ::testing::Values(ov::test::utils::LogicalTypes::LOGICAL_NOT),
        ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
        ::testing::Values(ov::element::bf16),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_Logical_Eltwise_CPU_BF16,
                         LogicalLayerCPUTest,
                         LogicalTestParams,
                         LogicalLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Logical_Not_Eltwise_CPU_BF16,
                         LogicalLayerCPUTest,
                         LogicalTestParamsNot,
                         LogicalLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
