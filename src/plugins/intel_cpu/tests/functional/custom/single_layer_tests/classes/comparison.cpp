// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison.hpp"

#include "internal_properties.hpp"
#include "utils/general_utils.h"
#include "utils/precision_support.h"
#include "common_test_utils/node_builders/comparison.hpp"

#if defined(OPENVINO_ARCH_RISCV64)
#   include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#endif

using namespace CPUTestUtils;
using namespace ov::test::utils;

namespace ov {
namespace test {
std::string ComparisonLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ComparisonLayerCPUTestParamSet> &obj) {
    const auto& [shapes, comparisonType, secondInType, modelPrc, inferPrc, enforceSnippets] = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "OpType=" << comparisonType << "_";
    result << "secondInType=" << secondInType << "_";
    result << "modelPrc=" << modelPrc.to_string() << "_";
    result << "inferPrc=" << inferPrc.to_string() << "_";
    result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

void ComparisonLayerCPUTest::SetUp() {
    const auto& [shapes, comparisonType, secondInType, modelPrc, inferPrc, enforceSnippets] = this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;

    // we have to change model precision as well, otherwise inference precision won't affect single-node graph
    // due to enforce inference precision optimization for the eltwise as first node of the model
    const auto currModelPrc = ov::element::Type(modelPrc).is_real() ? inferPrc : modelPrc;

    const auto primitiveType = getPrimitiveType(comparisonType, currModelPrc);
    const auto primiticePrcType =
        ov::intel_cpu::any_of(primitiveType, "ref", "acl") ? ov::element::f32.to_string() : currModelPrc.to_string();
    selectedType = primitiveType.empty() ? "" : primitiveType + "_" + primiticePrcType;

    init_input_shapes(shapes);

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }
    configuration.insert(ov::hint::inference_precision(inferPrc));

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(currModelPrc, inputDynamicShapes[0])};

    std::shared_ptr<ov::Node> secondInput;
    if (ov::test::utils::InputLayerType::CONSTANT == secondInType) {
        auto tensor = ov::test::utils::create_and_fill_tensor(currModelPrc, targetStaticShapes[0][1]);
        secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        auto param = std::make_shared<ov::op::v0::Parameter>(currModelPrc, inputDynamicShapes[1]);
        secondInput = param;
        params.push_back(param);
    }
    const auto comparison_node = ov::test::utils::make_comparison(params[0], secondInput, comparisonType);

    comparison_node->get_rt_info() = getCPUInfo();
    function = std::make_shared<ov::Model>(comparison_node, params, "Comparison");
}

std::string ComparisonLayerCPUTest::getPrimitiveType(const utils::ComparisonTypes& type, ov::element::Type modelType) const {
#if defined(OPENVINO_ARCH_ARM64)
    return "jit";
#endif
#if defined(OPENVINO_ARCH_ARM) && defined(OV_CPU_WITH_ACL)
    // TODO [171225] : On ARM there is ACL executor support which requires U8 on output.
    //                 This requirement is not met in Eltwise CPU node - ref impl is used.
    return modelType == ov::element::i32 ? "acl" : "ref";
#endif
#if defined(OPENVINO_ARCH_RISCV64)
    if (ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)) {
        if (ov::intel_cpu::any_of(type, utils::ComparisonTypes::EQUAL,
                                        utils::ComparisonTypes::NOT_EQUAL,
                                        utils::ComparisonTypes::LESS_EQUAL,
                                        utils::ComparisonTypes::GREATER_EQUAL))
            return "jit";
    }
#endif
    return CPUTestsBase::getPrimitiveType();
}

TEST_P(ComparisonLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

namespace comparison {

const std::vector<std::vector<InputShape>>& inShapesWithParameter() {
    static const std::vector<std::vector<InputShape>> shapes {
        {
            { {}, {{3, 33, 245}} },
            { {}, {{1, 33, 245}} },
        },
        {
            { {-1, -1}, {{3, 128}, {3, 128}, {5, 1},   {3, 128}} },
            { {-1, -1}, {{3, 128}, {1, 1},   {5, 128}, {3, 128}} }
        },
    };

    return shapes;
}

const std::vector<std::vector<InputShape>>& inShapesWithConstant() {
    static const std::vector<std::vector<InputShape>> shapes {
        {
            { {}, {{3, 33, 245}} },
            { {}, {{1, 1, 1}} },
        },
        {
            { {-1, -1}, {{5, 197}, {1, 197}, {1, 1}, {5, 197}} },
            { {}, {{1, 1}} }
        },
    };

    return shapes;
}

const std::vector<utils::ComparisonTypes>& comparisonTypes() {
    static const std::vector<utils::ComparisonTypes> types {
        utils::ComparisonTypes::EQUAL,
        utils::ComparisonTypes::NOT_EQUAL,
        utils::ComparisonTypes::GREATER,
        utils::ComparisonTypes::GREATER_EQUAL,
        utils::ComparisonTypes::LESS,
        utils::ComparisonTypes::LESS_EQUAL,
    };

    return types;
}

const std::vector<bool>& enforceSnippets() {
    static const std::vector<bool> enforce = {
#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_X86_64)
        true,
#endif
        false
    };

    return enforce;
}

const std::vector<ov::element::Type>& modelPrc() {
    static const std::vector<ov::element::Type> prc {
        ov::element::i32,
        ov::element::f32,
        ov::element::f16,
    };

    return prc;
}

const std::vector<ov::element::Type> inferPrc() {
    std::vector<ov::element::Type> prc {
        ov::element::f32
    };
    if (ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        prc.push_back(ov::element::f16);
    }
    if (ov::intel_cpu::hasHardwareSupport(ov::element::bf16)) {
        prc.push_back(ov::element::bf16);
    }

    return prc;
}

}  // namespace comparison

}  // namespace test
}  // namespace ov
