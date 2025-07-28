// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logical.hpp"

#include "internal_properties.hpp"
#include "common_test_utils/node_builders/logical.hpp"
#include "openvino/op/logical_not.hpp"

#if defined(OPENVINO_ARCH_RISCV64)
#   include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#endif

using namespace CPUTestUtils;
using namespace ov::test::utils;

namespace ov {
namespace test {
std::string LogicalLayerCPUTest::getTestCaseName(const testing::TestParamInfo<LogicalLayerCPUTestParamSet> &obj) {
    const auto [shapes, logicalType, secondInType, enforceSnippets] = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "OpType=" << logicalType << "_";
    if (ov::test::utils::LogicalTypes::LOGICAL_NOT != logicalType) {
        result << "secondInType=" << secondInType << "_";
    }
    result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

void LogicalLayerCPUTest::SetUp() {
    const auto [shapes, logicalType, secondInType, enforceSnippets] = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;

    const auto primitiveType = getPrimitiveType(logicalType);
    selectedType = primitiveType.empty() ? "" : primitiveType + "_I8";

    init_input_shapes(shapes);

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    const auto prc = ov::element::boolean;  // Because ngraph supports only boolean input for logical ops
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(prc, inputDynamicShapes[0])};

    std::shared_ptr<ov::Node> logical_node;
    if (ov::test::utils::LogicalTypes::LOGICAL_NOT != logicalType) {
        std::shared_ptr<ov::Node> secondInput;
        if (ov::test::utils::InputLayerType::CONSTANT == secondInType) {
            auto tensor = ov::test::utils::create_and_fill_tensor(prc, targetStaticShapes[0][1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        } else {
            auto param = std::make_shared<ov::op::v0::Parameter>(prc, inputDynamicShapes[1]);
            secondInput = param;
            params.push_back(param);
        }
        logical_node = ov::test::utils::make_logical(params[0], secondInput, logicalType);
    } else {
        logical_node = std::make_shared<ov::op::v1::LogicalNot>(params[0]);
    }

    logical_node->get_rt_info() = getCPUInfo();
    function = std::make_shared<ov::Model>(logical_node, params, "Logical");
}

std::string LogicalLayerCPUTest::getPrimitiveType(const utils::LogicalTypes& log_type) const {
#if defined(OPENVINO_ARCH_ARM64)
    return "jit";
#endif
#if defined(OPENVINO_ARCH_ARM)
    return "ref";
#endif
#if defined(OPENVINO_ARCH_RISCV64)
    if (ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)) {
        if ((activation_type == utils::LogicalTypes::LogicalAnd) ||
            (activation_type == utils::LogicalTypes::LogicalXor) ||
            (activation_type == utils::LogicalTypes::LogicalNot))
            return "jit";
    }
#endif
    return CPUTestsBase::getPrimitiveType();
}

TEST_P(LogicalLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

namespace logical {

const std::vector<std::vector<InputShape>>& inUnaryShapes() {
    static const std::vector<std::vector<InputShape>> shapes {
        {
            { {}, {{3, 33, 245}} }
        },
        {
            { {-1, -1}, {{3, 128}, {3, 128}, {5, 1}, {1, 128}} }
        },
    };

    return shapes;
}

const std::vector<std::vector<InputShape>>& inBinaryShapes() {
    static const std::vector<std::vector<InputShape>> shapes {
        {
            { {}, {{3, 33, 245}} },
            { {}, {{1, 33, 245}} },
        },
        {
            { {}, {{3, 33, 245}} },
            { {}, {{1, 1, 1}} },
        },
        {
            { {-1, -1}, {{3, 128}, {3, 128}, {5, 1},   {3, 128}} },
            { {-1, -1}, {{3, 128}, {1, 1},   {5, 128}, {3, 128}} }
        },
    };

    return shapes;
}

const std::vector<ov::test::utils::InputLayerType>& secondInTypes() {
    static const std::vector<ov::test::utils::InputLayerType> types {
        ov::test::utils::InputLayerType::PARAMETER,
    };

    return types;
}

const std::vector<utils::LogicalTypes>& logicalUnaryTypes() {
    static const std::vector<utils::LogicalTypes> types {
        utils::LogicalTypes::LOGICAL_NOT
    };

    return types;
}

const std::vector<utils::LogicalTypes>& logicalBinaryTypes() {
    static const std::vector<utils::LogicalTypes> types {
        utils::LogicalTypes::LOGICAL_AND,
        utils::LogicalTypes::LOGICAL_XOR,
        utils::LogicalTypes::LOGICAL_OR
    };

    return types;
}

const std::vector<bool>& enforceSnippets() {
    static const std::vector<bool> enforce = {
        true, false
    };

    return enforce;
}

}  // namespace logical

}  // namespace test
}  // namespace ov
