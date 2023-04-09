// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <tuple>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"
#include "snippets_transformations/enforce_precision.hpp"
#include "common_test_utils/common_utils.hpp"
#include "two_binary_ops_function.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace enforce_precision_test {

class DummyPrecisionSelection {
public:
    DummyPrecisionSelection(
        const std::set<std::vector<element::Type>>& supported_in_precisions1,
        const std::set<std::vector<element::Type>>& supported_in_precisions2) :
        supported_in_precisions1(supported_in_precisions1),
        supported_in_precisions2(supported_in_precisions2) {
    }

    std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& op) noexcept {
        if (ov::is_type<ov::test::snippets::DummyOperation1>(op)) {
            return supported_in_precisions1;
        } else if (ov::is_type<ov::test::snippets::DummyOperation2>(op)) {
            return supported_in_precisions2;
        }
        return {};
    }

private:
    const std::set<std::vector<element::Type>> supported_in_precisions1;
    const std::set<std::vector<element::Type>> supported_in_precisions2;
};

class EnforcePrecisionParamsValues {
public:
    class Branch {
    public:
        Branch() = default;
        Branch(const element::Type& type, const size_t branches_amount = 1ul) :
            type(type), branches_amount(branches_amount) {}

        element::Type type;
        size_t branches_amount;
    };

    class Actual {
    public:
        Actual() = default;
        Actual(
            const std::pair<element::Type, element::Type>& convertion_before_op1,
            const element::Type& convertion_before_op2_1,
            const std::pair<element::Type, element::Type>& convertion_before_op2_2,
            const std::vector<Branch>& convertion_after_op2,
            const std::set<std::vector<element::Type>>& supported_precisions1,
            const std::set<std::vector<element::Type>>& supported_precisions2,
            const std::map<std::vector<element::Type>, std::vector<element::Type>>& op_precisions1,
            const std::map<std::vector<element::Type>, std::vector<element::Type>>& op_precisions2) :
            convertion_before_op1(convertion_before_op1),
            convertion_before_op2_1(convertion_before_op2_1),
            convertion_before_op2_2(convertion_before_op2_2),
            convertion_after_op2(convertion_after_op2),
            supported_precisions1(supported_precisions1),
            supported_precisions2(supported_precisions2),
            op_precisions1(op_precisions1),
            op_precisions2(op_precisions2) {
        }

        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        std::vector<Branch> convertion_after_op2;
        // supported precisions for enforcement
        std::set<std::vector<element::Type>> supported_precisions1;
        std::set<std::vector<element::Type>> supported_precisions2;
        // supported operation precisions for operation validation
        std::map<std::vector<element::Type>, std::vector<element::Type>> op_precisions1;
        std::map<std::vector<element::Type>, std::vector<element::Type>> op_precisions2;
    };

    class Expected {
    public:
        Expected() = default;
        Expected(
            const std::pair<element::Type, element::Type>& convertion_before_op1,
            const element::Type& convertion_before_op2_1,
            const std::pair<element::Type, element::Type>& convertion_before_op2_2,
            const std::vector<Branch>& convertion_after_op2,
            const element::Type& convertion_before_result) :
            convertion_before_op1(convertion_before_op1),
            convertion_before_op2_1(convertion_before_op2_1),
            convertion_before_op2_2(convertion_before_op2_2),
            convertion_after_op2(convertion_after_op2),
            convertion_before_result(convertion_before_result) {
        }

        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        std::vector<Branch> convertion_after_op2;
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

        auto supported_to_string = [](const std::set<std::vector<element::Type>>& supported_precisions) noexcept {
            std::ostringstream result;
            result << "{";
            for (const auto& precisions : supported_precisions) {
                result << "prc=" << CommonTestUtils::vec2str(precisions) << "_";
            }
            result << "}";
            return result.str();
        };

        auto op_to_string = [](const std::map<std::vector<element::Type>, std::vector<element::Type>>& op_precisions) noexcept {
            std::ostringstream result;
            result << "{";
            for (const auto& precisions : op_precisions) {
                result << "in_prc=" << CommonTestUtils::vec2str(precisions.first) << "_";
                result << "out_prc=" << CommonTestUtils::vec2str(precisions.second) << "_";
            }
            result << "}";
            return result.str();
        };

        auto branches_to_string = [](const std::vector<EnforcePrecisionParamsValues::Branch>& op_precisions) noexcept {
            std::ostringstream result;
            result << "{";
            for (const auto& branch : op_precisions) {
                result << "type=" << branch.type << "_";
                result << "branches_amount=" << branch.branches_amount << "_";
            }
            result << "}";
            return result.str();
        };

        std::ostringstream result;
        result << "in0=" << shapes.first << "_" << test_values.input_types[0] << "_"
            << "in1=" << shapes.second << "_" << test_values.input_types[1] << "_"
            << "in2=" << test_values.input_types[2] << "_"
            << "supported_precisions1=" << supported_to_string(test_values.actual.supported_precisions1) << "_"
            << "supported_precisions2=" << supported_to_string(test_values.actual.supported_precisions2) << "_"
            << "op_precisions1=" << op_to_string(test_values.actual.op_precisions1) << "_"
            << "op_precisions2=" << op_to_string(test_values.actual.op_precisions2) << "_"
            << test_values.expected.convertion_before_op1.first << "_" << test_values.expected.convertion_before_op1.second << "_"
            << test_values.expected.convertion_before_op2_1 << "_"
            << test_values.expected.convertion_before_op2_2.first << "_" << test_values.expected.convertion_before_op2_2.second << "_"
            << "branches=" << branches_to_string(test_values.expected.convertion_after_op2);
        return result.str();
    }
};

TEST_P(EnforcePrecisionTest, CompareFunctions) {
    disable_rt_info_check();

    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    const auto to_branches = [](const std::vector<EnforcePrecisionParamsValues::Branch>& test_branches) {
        std::vector<TwoBinaryOpsFunction::Branch> branches;
        for (const auto& test_branch : test_branches) {
            branches.push_back({test_branch.type, test_branch.branches_amount});
        }
        return branches;
    };

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
            to_branches(test_values.actual.convertion_after_op2),
            test_values.actual.op_precisions1,
            test_values.actual.op_precisions2,
        },
        {
            test_values.expected.convertion_before_op1,
            test_values.expected.convertion_before_op2_1,
            test_values.expected.convertion_before_op2_2,
            to_branches(test_values.expected.convertion_after_op2),
            test_values.expected.convertion_before_result
        });
    function = function_stub.getOriginal();

    auto dummyPrecisionSelection = std::make_shared<DummyPrecisionSelection>(
        test_values.actual.supported_precisions1,
        test_values.actual.supported_precisions2);

    auto get_supported_precisions = [dummyPrecisionSelection](const std::shared_ptr<ngraph::Node>& op) {
        return dummyPrecisionSelection->get_supported_precisions(op);;
    };

    manager.register_pass<ov::intel_cpu::pass::EnforcePrecision>(
        test_values.source,
        test_values.target,
        get_supported_precisions);

    function_ref = function_stub.getReference();

    // issue #108208
    if ((test_values.actual.convertion_after_op2.size() > 1ull) ||
        (!test_values.actual.convertion_after_op2.empty() && test_values.actual.convertion_after_op2[0].branches_amount > 1ull) ||
        (test_values.expected.convertion_after_op2.size() > 1ull) ||
        (!test_values.expected.convertion_after_op2.empty() && test_values.expected.convertion_after_op2[0].branches_amount > 1ull)) {
        comparator.disable(FunctionsComparator::CmpValues::GRAPH);
    }
}

std::vector<std::pair<PartialShape, PartialShape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

std::vector<EnforcePrecisionParamsValues> test_values {
    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // supported precisions
            // operation #1 supports 2 inputs in bf16 - will be converted
            {{element::bf16, element::bf16}},
            // operation #2 supports 3 inputs in bf16 - will be not converted: has 2 inputs
            {{element::bf16, element::bf16, element::bf16}},
            // operation in/out precisions
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}},
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}},
            }
        },
        {
            // convertions after
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // supported precisions
            // operation #1 supports 2 inputs in bf16 - will be converted
            {{element::bf16, element::bf16}},
            // operation #2 doesn't configured to support bf16 - will be not converted
            {},
            // operation in/out precisions
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}},
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}},
            }
        },
        {
            // convertions after
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            {{element::bf16, element::bf16}},
            {{element::bf16, element::bf16}},
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {element::undefined, element::bf16},
            {},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::bf16}},
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::f32}} // <= operation #2 returns FP32
            },
        },
        {
            // convertions after
            {},
            {},
            {element::undefined, element::bf16},
            {},
            {element::bf16}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::bf16}},
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::f32}} // <= operation #1 returns FP32
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {element::bf16, element::bf16},
            {},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32}, // <= operation #2 has f32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::f32}}, // <= operation #2 supports f32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::f32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32}, // <= operation #2 has f32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16, element::undefined, element::undefined}, // <= after operation #2 there are 3 branches
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::f32}}, // <= operation #2 supports f32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::f32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {element::undefined, {element::bf16, 2}},
            {}
        }
    },

    //  Input1   Input2
    //     \      /
    //     Operation1  Constant
    //          \      /
    //         Operation2
    //           /    \
    //       Result  Convert <= can not be removed
    //                  \
    //                Result
    //
    {
        {element::bf16, element::bf16, element::f32}, // <= operation #2 has f32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::undefined, {element::bf16, 0}}, // <= after operation #2 there are 2 branches
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::f32}}, // <= operation #2 supports f32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::f32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {element::f32, {element::bf16, 0}},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32}, // <= operation #2 has f32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::undefined, {element::bf16, 2}}, // <= after operation #2 convertion there are 2 branches
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::f32}}, // <= operation #2 supports f32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::f32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {element::f32, {element::undefined, 2}},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32}, // <= operation #2 has f32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16, element::undefined}, // <= after operation #2 there are 2 branches
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::f32}}, // <= operation #2 supports f32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::f32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {element::undefined, element::bf16},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::i32}, // <= operation #2 has i32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations support BF16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::bf16}}, // <= operation #2 supports bf16 (not i32) on the second input: i32 can not be converted to bf16
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::i32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {element::f32, element::undefined},
            {},
            {element::bf16}
        }
    },

    {
        {element::bf16, element::bf16, element::i32}, // <= operation #2 has i32 on the second input
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations support bf16
            {{element::bf16, element::bf16}},
            {{element::bf16, element::i32}}, // <= operation #2 supports i32 on the second input
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::i32}, {element::f32}},
                {{element::bf16, element::i32}, {element::bf16}}
            },
        },
        {
            // convertions after
            {},
            {},
            {},
            {},
            {}
        }
    },

    {
        {element::bf16, element::bf16, element::f32},
        element::f32,
        element::bf16,
        {
            // convertions before
            {element::f32, element::f32},
            {},
            {},
            {element::bf16},
            // both operations are configured not support bf16
            {},
            {},
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
            {
                {{element::f32, element::f32}, {element::f32}},
                {{element::bf16, element::bf16}, {element::bf16}}
            },
        },
        {
            // convertions after: nothing was changed
            {element::f32, element::f32},
            {},
            {},
            {},
            {element::bf16}
        }
    }
};

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
