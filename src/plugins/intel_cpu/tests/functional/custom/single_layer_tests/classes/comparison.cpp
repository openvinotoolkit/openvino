// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison.hpp"

#include <array>
#include <cstdint>
#include <limits>

#include "common_test_utils/node_builders/comparison.hpp"
#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/logical_not.hpp"
#include "utils/general_utils.h"
#include "utils/precision_support.h"

#if defined(OPENVINO_ARCH_RISCV64)
#   include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#endif

using namespace CPUTestUtils;
using namespace ov::test::utils;

namespace ov {
namespace test {
namespace {

ov::Tensor generate_i32_comparison_input(const ov::Shape& shape, size_t port, ComparisonGraphType graphType) {
    constexpr int32_t limit = 1 << 24;
    constexpr int32_t min = std::numeric_limits<int32_t>::min();
    constexpr int32_t max = std::numeric_limits<int32_t>::max();
    // Keep adjacent integers in each window: converting them to f32 can make unequal inputs compare equal.
    constexpr std::array<std::array<int32_t, 2>, 14> values = {{{limit + 1, limit},
                                                                {limit, limit + 1},
                                                                {limit + 1, limit + 1},
                                                                {-limit - 1, -limit},
                                                                {-limit, -limit - 1},
                                                                {-limit - 1, -limit - 1},
                                                                {min, min + 1},
                                                                {min + 1, min},
                                                                {max, max - 1},
                                                                {max - 1, max},
                                                                {min, max},
                                                                {max, min},
                                                                {0, 0},
                                                                {-1, 1}}};
    // Add consumes exactly representable values, while the other comparison input is an adjacent integer.
    // This isolates comparison rounding by keeping the Add operands and result exactly representable in f32.
    constexpr std::array<std::array<int32_t, 2>, 6> addValues = {{{limit + 4, limit + 3},
                                                                  {limit + 4, limit + 5},
                                                                  {-limit - 4, -limit - 3},
                                                                  {-limit - 4, -limit - 5},
                                                                  {limit, limit + 1},
                                                                  {-limit, -limit - 1}}};
    ov::Tensor tensor(ov::element::i32, shape);
    auto* data = tensor.data<int32_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = graphType == ComparisonGraphType::ADD_COMPARE
                      ? addValues[i % addValues.size()][port] - (port == 0 ? 2 : 0)
                      : values[i % values.size()][port];
    }
    return tensor;
}

}  // namespace

std::string ComparisonLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ComparisonLayerCPUTestParamSet> &obj) {
    const auto& [shapes, comparisonType, secondInType, modelPrc, inferPrc, enforceSnippets, largeI32Values, graphType] =
        obj.param;

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
    result << "_largeI32Values=" << largeI32Values;
    result << "_graphType="
           << (graphType == ComparisonGraphType::COMPARE               ? "Compare"
               : graphType == ComparisonGraphType::COMPARE_LOGICAL_NOT ? "CompareLogicalNot"
                                                                       : "AddCompare");

    return result.str();
}

void ComparisonLayerCPUTest::SetUp() {
    const auto& [shapes, comparisonType, secondInType, modelPrc, inferPrc, enforceSnippets, largeI32Values, graphType] =
        this->GetParam();
    targetDevice = ov::test::utils::DEVICE_CPU;
    i32Window = window;

    // we have to change model precision as well, otherwise inference precision won't affect single-node graph
    // due to enforce inference precision optimization for the eltwise as first node of the model
    const auto currModelPrc = ov::element::Type(modelPrc).is_real() ? inferPrc : modelPrc;
    OPENVINO_ASSERT(!largeI32Values || currModelPrc == ov::element::i32);

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
        auto tensor = largeI32Values ? generate_i32_comparison_input(targetStaticShapes[0][1], 1, graphType)
                                     : ov::test::utils::create_and_fill_tensor(currModelPrc, targetStaticShapes[0][1]);
        secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        auto param = std::make_shared<ov::op::v0::Parameter>(currModelPrc, inputDynamicShapes[1]);
        secondInput = param;
        params.push_back(param);
    }
    std::shared_ptr<ov::Node> firstInput = params[0];
    if (graphType == ComparisonGraphType::ADD_COMPARE) {
        firstInput =
            std::make_shared<ov::op::v1::Add>(firstInput, ov::op::v0::Constant::create(currModelPrc, ov::Shape{}, {2}));
    }
    auto comparison_node = ov::test::utils::make_comparison(firstInput, secondInput, comparisonType);
    if (graphType == ComparisonGraphType::COMPARE_LOGICAL_NOT) {
        comparison_node = std::make_shared<ov::op::v1::LogicalNot>(comparison_node);
    }
    if (graphType != ComparisonGraphType::COMPARE) {
        // Neighboring operations may use a different execution precision.
        selectedType = primitiveType + "_.*";
    }

    function = create_ov_model(currModelPrc, params, comparison_node, "Comparison");
}

void ComparisonLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    if (!std::get<6>(GetParam())) {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        return;
    }

    inputs.clear();
    const auto& parameters = function->get_parameters();
    for (size_t i = 0; i < parameters.size(); ++i) {
        inputs.emplace(parameters[i],
                       generate_i32_comparison_input(targetInputStaticShapes[i], i, std::get<7>(GetParam())));
    }
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
                                        utils::ComparisonTypes::LESS,
                                        utils::ComparisonTypes::LESS_EQUAL,
                                        utils::ComparisonTypes::GREATER,
                                        utils::ComparisonTypes::GREATER_EQUAL))
            return "jit";
    }
#endif
    return CPUTestsBase::getPrimitiveType();
}

TEST_P(ComparisonLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
    if (std::get<6>(GetParam())) {
        const bool enforceSnippets = std::get<5>(GetParam());
        const bool logicalNot = std::get<7>(GetParam()) == ComparisonGraphType::COMPARE_LOGICAL_NOT;
        CheckNumberOfNodesWithType(compiledModel,
                                   enforceSnippets ? "Subgraph" : "Eltwise",
                                   enforceSnippets || !logicalNot ? 1 : 2);
    }
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
