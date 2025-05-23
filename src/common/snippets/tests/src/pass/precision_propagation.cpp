// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/precision_propagation.hpp"

#include <gtest/gtest.h>

#include "snippets/lowered/expression.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "common_test_utils/common_utils.hpp"
#include "precision_propagation.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

class DummyPrecisionPropagationTargetMachine : public DummyTargetMachine {
public:
    DummyPrecisionPropagationTargetMachine(
        const std::set<std::vector<element::Type>>& op1_supported_precisions,
        const std::set<std::vector<element::Type>>& op2_supported_precisions)
        : DummyTargetMachine() {
        jitters[DummyAdd::get_type_info_static()] = ov::snippets::jitters_value {
            [](const ov::snippets::lowered::ExpressionPtr& n) { return std::make_shared<DummyEmitter>(); },
            [op1_supported_precisions](const std::shared_ptr<ov::Node>& n) { return op1_supported_precisions; }};
        jitters[op::v1::Maximum::get_type_info_static()] = ov::snippets::jitters_value{
            [](const ov::snippets::lowered::ExpressionPtr& n) { return std::make_shared<DummyEmitter>(); },
            [op2_supported_precisions](const std::shared_ptr<ov::Node>&n) { return op2_supported_precisions; }};

        auto default_jitter = ov::snippets::jitters_value{
            [](const ov::snippets::lowered::ExpressionPtr& n) { return std::make_shared<DummyEmitter>(); },
            [](const std::shared_ptr<ov::Node>& n) { return std::set<std::vector<element::Type>>{};} };
        jitters[ov::snippets::op::ConvertSaturation::get_type_info_static()] = default_jitter;
    }
};

} // namespace

std::string PrecisionPropagationTest::getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj) {
    std::pair<PartialShape, PartialShape> shapes;
    PrecisionPropagationParamsValues test_values;
    std::tie(shapes, test_values) = obj.param;

    auto to_string = [](const std::set<std::vector<element::Type>>& precisions_pack) noexcept {
        std::ostringstream result;
        result << "{";
        for (const auto& precisions : precisions_pack) {
            result << ov::test::utils::vec2str(precisions) << "_";
        }
        result << "}";
        return result.str();
    };

    std::ostringstream result;
    result << "IN0_" << shapes.first << "_" << test_values.input_types[0] << "_"
           << "IN1_" << shapes.second << "_" << test_values.input_types[1] << "_"
           << "IN2_" << test_values.input_types[2]
           << to_string(test_values.actual.op1_supported_precisions) << "_"
           << to_string(test_values.actual.op2_supported_precisions) << "_"
           << test_values.expected.convertion_before_op1.first << "_" << test_values.expected.convertion_before_op1.second << "_"
           << test_values.expected.convertion_before_op2_1 << "_"
           << test_values.expected.convertion_before_op2_2.first << "_" << test_values.expected.convertion_before_op2_2.second << "_"
           << test_values.expected.convertion_after_op2 << "_";
    return result.str();
}

TEST_P(PrecisionPropagationTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    const auto input_shapes = std::vector<PartialShape>({ shapes.first, shapes.second });
    PrecisionPropagationAddFunction function_stub(
        input_shapes,
        test_values.input_types[0],
        test_values.input_types[1],
        test_values.input_types[2],
        {
            test_values.actual.convertion_before_op1,
            test_values.actual.convertion_before_op2_1,
            test_values.actual.convertion_before_op2_2
        },
        {
            test_values.expected.convertion_before_op1,
            test_values.expected.convertion_before_op2_1,
            test_values.expected.convertion_before_op2_2,
            test_values.expected.convertion_after_op2
        });
    model = function_stub.getOriginal();

    const auto target_machine = std::make_shared<DummyPrecisionPropagationTargetMachine>(
        test_values.actual.op1_supported_precisions,
        test_values.actual.op2_supported_precisions);

    manager.register_pass<ov::snippets::pass::PropagatePrecision>(target_machine);

    model_ref = function_stub.getReference();
}

namespace PrecisionPropagationTestInstantiation {
// clang-format off

std::vector<std::pair<PartialShape, PartialShape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<PrecisionPropagationParamsValues> test_cases {
    {
        {element::f32, element::f32, element::f32},
        {
            {},
            {},
            {},
            {{element::f32, element::f32}},
            {{element::f32, element::f32}}
        },
        {}
    },
    // in:  Parameter I8 => Op1 I32 => Convert I8 => Op1 I8 => Result
    // out: Parameter I8 => Add I32 => Convert I8 => Convert FP32 => Op1 FP32 => Result
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {},
            {{element::i8, element::i8}},
            {{element::f32, element::f32}}
        },
        {
            {},
            element::i8,
            {element::f32, element::f32},
            {element::i8}
        }
    },
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {},
            {{element::i8, element::i8}},
            {{element::i8, element::i8}}
        },
        {
            {},
            {},
            {element::i8, element::dynamic},
            {}
        }
    },
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {},
            {{element::i8, element::i8}},
            {{element::i32, element::i32}}
        },
        {
            {},
            {element::i8},
            {element::i32, element::i32},
            {element::i8}
        }
    },
    {
        {element::bf16, element::bf16, element::f32},
        {
            {element::f32, element::f32},
            {},
            {},
            {
                {element::f32, element::f32},
                {element::i8, element::i8}
            },
            {
                {element::f32, element::f32},
                {element::i32, element::i32}
            }
        },
        {
            {element::f32, element::f32},
            {},
            {},
            {}
        }
    },
    // propagate precision via operation #1
    {
        {element::bf16, element::bf16, element::f32},
        {
            {element::f32, element::f32},
            {},
            {},
            {
                {element::f32, element::f32},
                {element::bf16, element::bf16}
            },
            {
                {element::f32, element::f32}
            }
        },
        {
            {},
            {},
            {element::f32, element::dynamic},
            {}
        }
    },
    // propagate precision via operation #1
    {
        {element::bf16, element::bf16, element::bf16},
        {
            {element::f32, element::f32},
            {},
            {element::dynamic, element::f32},
            {
                {element::f32, element::f32},
                {element::bf16, element::bf16}
            },
            {
                {element::f32, element::f32}
            }
        },
        {
            {},
            {},
            {element::f32, element::f32},
            {}
        }
    },
    // propagate precision via both operations
    {
        {element::bf16, element::bf16, element::bf16},
        {
            {element::f32, element::f32},
            {},
            {element::dynamic, element::f32},
            {
                {element::f32, element::f32},
                {element::bf16, element::bf16}
            },
            {
                {element::f32, element::f32},
                {element::bf16, element::bf16}
            }
        },
        {
            {},
            {},
            {},
            {element::f32}
        }
    },
    {
        {element::bf16, element::bf16, element::bf16},
        {
            {},
            {},
            {},
            {{element::f32, element::f32}},
            {{element::f32, element::f32}}
        },
        {
            {{element::f32}, {element::f32}},
            {element::bf16},
            {{element::f32}, {element::f32}},
            {element::bf16}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_PrecisionPropagationTest,
    PrecisionPropagationTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(test_cases)),
    PrecisionPropagationTest::getTestCaseName);

// clang-format on
} // namespace PrecisionPropagationTestInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov
