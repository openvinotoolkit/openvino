// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <tuple>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "common_test_utils/common_utils.hpp"
#include "two_binary_ops.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace enforce_precision_test {

class DummyPrecisionSelection {
public:
    DummyPrecisionSelection(
        const std::set<std::vector<element::Type>>& precisions1,
        const std::set<std::vector<element::Type>>& precisions2) : precisions1(precisions1), precisions2(precisions2) {
    }

    std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& op) noexcept {
        if (ov::is_type<ov::test::snippets::DummyOperation1>(op)) {
            return precisions1;
        } else if (ov::is_type<ov::test::snippets::DummyOperation2>(op)) {
            return precisions2;
        }
        return {};
    }

private:
    const std::set<std::vector<element::Type>> precisions1;
    const std::set<std::vector<element::Type>> precisions2;
};

class EnforcePrecisionParamsValues {
public:
    class Actual {
    public:
        Actual() = default;
        Actual(
            std::pair<element::Type, element::Type> convertion_before_op1,
            element::Type convertion_before_op2_1,
            std::pair<element::Type, element::Type> convertion_before_op2_2,
            element::Type convertion_after_op2,
            std::set<std::vector<element::Type>> precisions1,
            std::set<std::vector<element::Type>> precisions2) :
            convertion_before_op1(convertion_before_op1),
            convertion_before_op2_1(convertion_before_op2_1),
            convertion_before_op2_2(convertion_before_op2_2),
            convertion_after_op2(convertion_after_op2),
            precisions1(precisions1),
            precisions2(precisions2) {
        }

        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
        std::set<std::vector<element::Type>> precisions1;
        std::set<std::vector<element::Type>> precisions2;
    };

    class Expected {
    public:
        Expected() = default;
        Expected(
            std::pair<element::Type, element::Type> convertion_before_op1,
            element::Type convertion_before_op2_1,
            std::pair<element::Type, element::Type> convertion_before_op2_2,
            element::Type convertion_after_op2,
            element::Type convertion_before_result) :
            convertion_before_op1(convertion_before_op1),
            convertion_before_op2_1(convertion_before_op2_1),
            convertion_before_op2_2(convertion_before_op2_2),
            convertion_after_op2(convertion_after_op2),
            convertion_before_result(convertion_before_result) {
        }

        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
        element::Type convertion_before_result;
    };

    EnforcePrecisionParamsValues() = default;
    EnforcePrecisionParamsValues(
        std::vector<element::Type> input_types,
        element::Type source,
        element::Type target,
        Actual actual,
        Expected expected) :
        input_types(input_types), source(source), target(target), actual(actual), expected(expected) {
    }

    std::vector<element::Type> input_types;
    element::Type source;
    element::Type target;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    std::pair<PartialShape, PartialShape>, // input shapes
    EnforcePrecisionParamsValues
> EnforcePrecisionParams;

class EnforcePrecisionTest : public TransformationTestsF,
                             public testing::WithParamInterface<EnforcePrecisionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EnforcePrecisionParams> obj) {
        std::pair<PartialShape, PartialShape> shapes;
        EnforcePrecisionParamsValues test_values;
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
        result << "in0=" << shapes.first << "_" << test_values.input_types[0] << "_"
            << "in1=" << shapes.second << "_" << test_values.input_types[1] << "_"
            << "in2=" << test_values.input_types[2] << "_"
            << "_precisions1=" << to_string(test_values.actual.precisions1) << "_"
            << "_precisions2=" << to_string(test_values.actual.precisions2) << "_"
            << test_values.expected.convertion_before_op1.first << "_" << test_values.expected.convertion_before_op1.second << "_"
            << test_values.expected.convertion_before_op2_1 << "_"
            << test_values.expected.convertion_before_op2_2.first << "_" << test_values.expected.convertion_before_op2_2.second << "_"
            << test_values.expected.convertion_after_op2 << "_";
        return result.str();
    }
};

TEST_P(EnforcePrecisionTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    const auto input_shapes = std::vector<PartialShape>({ shapes.first, shapes.second });
    TwoBinaryOpsFunction function_stub(
        input_shapes,
        test_values.input_types[0],
        test_values.input_types[1],
        test_values.input_types[2],
        {
            test_values.actual.convertion_before_op1,
            test_values.actual.convertion_before_op2_1,
            test_values.actual.convertion_before_op2_2,
            test_values.actual.convertion_after_op2
        },
        {
            test_values.expected.convertion_before_op1,
            test_values.expected.convertion_before_op2_1,
            test_values.expected.convertion_before_op2_2,
            test_values.expected.convertion_after_op2,
            test_values.expected.convertion_before_result
        });
    model = function_stub.getOriginal();

    auto dummyPrecisionSelection = std::make_shared<DummyPrecisionSelection>(test_values.actual.precisions1, test_values.actual.precisions2);

    auto get_supported_precisions = [dummyPrecisionSelection](const std::shared_ptr<ov::Node>& op) {
        return dummyPrecisionSelection->get_supported_precisions(op);;
    };

    manager.register_pass<ov::intel_cpu::pass::EnforcePrecision>(
        test_values.source,
        test_values.target,
        get_supported_precisions);

    model_ref = function_stub.getReference();
}

std::vector<std::pair<PartialShape, PartialShape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<EnforcePrecisionParamsValues> test_values{
    {{element::bf16, element::bf16, element::f32},
     element::f32,
     element::bf16,
     {{element::f32, element::f32},
      {},
      {},
      {element::bf16},
      {
          {{element::bf16, element::bf16, element::bf16}, {element::bf16, element::bf16}},
      },
      {{{element::bf16, element::bf16, element::bf16}}}},
     {{}, {}, {element::f32, element::dynamic}, {}, {element::bf16}}},

    {{element::bf16, element::bf16, element::f32},
     element::f32,
     element::bf16,
     {{element::f32, element::f32},
      {},
      {},
      {element::bf16},
      {
          {{element::bf16, element::bf16}},
      },
      {{{element::bf16, element::bf16}}}},
     {{}, {}, {element::dynamic, element::bf16}, {element::f32}, {element::bf16}}},

    {{element::bf16, element::bf16, element::f32},
     element::f32,
     element::bf16,
     {{element::f32, element::f32},
      {},
      {},
      {element::bf16},
      {
          {{element::bf16, element::bf16}},
      },
      {{{element::bf16, element::f32}}}},
     {{}, {}, {}, {element::f32}, {element::bf16}}},

    {{element::bf16, element::bf16, element::i32},
     element::f32,
     element::bf16,
     {{element::f32, element::f32},
      {},
      {},
      {element::bf16},
      {
          {{element::bf16, element::bf16}},
      },
      {{{element::bf16, element::bf16}}}},
     {{}, {}, {element::f32, element::dynamic}, {}, {element::bf16}}},

    {{element::bf16, element::bf16, element::i32},
     element::f32,
     element::bf16,
     {{element::f32, element::f32},
      {},
      {},
      {element::bf16},
      {
          {{element::bf16, element::bf16}},
      },
      {{{element::bf16, element::i32}}}},
     {{}, {}, {}, {element::f32}, {element::bf16}}},

    {{element::f16, element::f16, element::f32},
     element::f32,
     element::f16,
     {{element::f32, element::f32},
      {},
      {},
      {element::f16},
      {
          {{element::f16, element::f16}},
      },
      {{{element::f16, element::f32}}}},
     {{}, {}, {}, {element::f32}, {element::f16}}},

    {{element::f16, element::f16, element::f32},
     element::f32,
     element::f16,
     {{element::f32, element::f32}, {}, {}, {element::f16}, {}, {}},
     {{element::f32, element::f32}, {}, {}, {}, {element::f16}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_EnforcePrecisionTest,
    EnforcePrecisionTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(test_values)),
    EnforcePrecisionTest::getTestCaseName);

} // namespace enforce_precision_test

}  // namespace snippets
}  // namespace test
}  // namespace ov
