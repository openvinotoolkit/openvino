// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/precision_propagation.hpp"

#include <gtest/gtest.h>
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "common_test_utils/common_utils.hpp"
#include "precision_propagation_function.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
class DummyAdd : public ngraph::opset1::Add {
public:
    OPENVINO_OP("DummyAdd", "test::snippets");

    DummyAdd(const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast =
            ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : ngraph::opset1::Add(arg0, arg1, auto_broadcast) {
        constructor_validate_and_infer_types();
    }

    DummyAdd(const ngraph::opset1::Add& add)
        : Add(add.get_input_source_output(0), add.get_input_source_output(1), add.get_autob()) {
        constructor_validate_and_infer_types();
    }

    DummyAdd() = default;

    void validate_and_infer_types() override {
        const auto input_type1 = get_input_element_type(0);
        const auto input_type2 = get_input_element_type(1);

        const element::Type output_type = (input_type1 == element::i8) || (input_type2 == element::i8) ?
            element::i32 :
            get_input_element_type(0);

        set_output_type(0, output_type, get_input_partial_shape(0));
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<DummyAdd>(new_args.at(0), new_args.at(1), this->get_autob());
    }
};

class DummyPrecisionPropogationTargetMachine : public DummyTargetMachine {
public:
    DummyPrecisionPropogationTargetMachine(
        const std::set<std::vector<InferenceEngine::Precision>>& op1_supported_precisions,
        const std::set<std::vector<InferenceEngine::Precision>>& op2_supported_precisions)
        : DummyTargetMachine() {
        jitters[DummyAdd::get_type_info_static()] = ngraph::snippets::jitters_value {
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            op1_supported_precisions};
        jitters[op::v1::Maximum::get_type_info_static()] = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            op2_supported_precisions};

        auto default_jitter = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            {}};
        jitters[ngraph::snippets::op::ConvertSaturation::get_type_info_static()] = default_jitter;
    }
};

} // namespace

std::string PrecisionPropagationTest::getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj) {
    std::pair<Shape, Shape> shapes;
    PrecisionPropagationParamsValues test_values;
    std::tie(shapes, test_values) = obj.param;

    auto to_string = [](const std::set<std::vector<InferenceEngine::Precision>>& precisions_pack) noexcept {
        std::ostringstream result;
        for (const auto& precisions : precisions_pack) {
            result << precisions << "_";
        }
        return result.str();
    };

    std::ostringstream result;
    result << "IN0_" << shapes.first << "_" << test_values.input_types[0] << "_"
           << "IN1_" << shapes.second << "_" << test_values.input_types[1] << "_"
           << "IN2_" << test_values.input_types[2]
           << to_string(test_values.actual.op1_supported_precisions) << "_"
           << to_string(test_values.actual.op2_supported_precisions) << "_"
           << test_values.expected.convertion_before_op1.first << "_" << test_values.expected.convertion_before_op1.second << "_"
           << test_values.expected.convertion_before_op2.first << "_" << test_values.expected.convertion_before_op2.second << "_"
           << test_values.expected.convertion_after_op2 << "_";
    return result.str();
}

TEST_P(PrecisionPropagationTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    function = PrecisionPropagationFunction::get<DummyAdd>(test_values.input_types[0],
                                                           shapes.first,
                                                           test_values.input_types[1],
                                                           shapes.second,
                                                           test_values.input_types[2],
                                                           test_values.actual.convertion_before_op1,
                                                           test_values.actual.convertion_before_op2);

    const auto target_machine =
        std::make_shared<DummyPrecisionPropogationTargetMachine>(test_values.actual.op1_supported_precisions,
                                                                 test_values.actual.op2_supported_precisions);
    ngraph::snippets::pass::PropagatePrecision(element::f32, target_machine).run_on_model(function);
    function->validate_nodes_and_infer_types();

    function_ref = PrecisionPropagationFunction::get<DummyAdd>(test_values.input_types[0],
                                                               shapes.first,
                                                               test_values.input_types[1],
                                                               shapes.second,
                                                               test_values.input_types[2],
                                                               test_values.expected.convertion_before_op1,
                                                               test_values.expected.convertion_before_op2,
                                                               test_values.expected.convertion_after_op2);
}

namespace PrecisionPropagationTestInstantiation {
// clang-format off

std::vector<std::pair<Shape, Shape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<PrecisionPropagationParamsValues> test_cases {
    {
        {element::f32, element::f32, element::f32},
        {
            {},
            {},
            {{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}},
            {{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}}
        },
        {}
    },
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {{InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}},
            {{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}}
        },
        {
            {},
            {element::f32, element::f32},
            {element::i8}
        }
    },
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {{InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}},
            {{InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}}
        },
        {
            {},
            {element::i8, element::undefined},
            {}
        }
    },
    {
        {element::i8, element::i8, element::i8},
        {
            {},
            {},
            {{InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}},
            {{InferenceEngine::Precision::I32, InferenceEngine::Precision::I32}}
        },
        {
            {},
            {element::undefined, element::i32},
            {element::i8}
        }
    },
    {
        {element::bf16, element::bf16, element::f32},
        {
            {element::f32, element::f32},
            {},
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}
            },
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::I32, InferenceEngine::Precision::I32}
            }
        },
        {
            {element::f32, element::f32},
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
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16}
            },
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}
            }
        },
        {
            {},
            {element::f32, element::undefined},
            {}
        }
    },
    // propagate precision via operation #1
    {
        {element::bf16, element::bf16, element::bf16},
        {
            {element::f32, element::f32},
            {element::undefined, element::f32},
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16}
            },
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}
            }
        },
        {
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
            {element::undefined, element::f32},
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16}
            },
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16}
            }
        },
        {
            {},
            {},
            {element::f32}
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
