// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/graph_comparator.hpp"
#include "snippets_transformations/remove_converts.hpp"
#include "snippets/op/convert_saturation.hpp"

namespace ov {
namespace test {
namespace snippets {

class RemoveConvertsFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const PartialShape& input_shape,
        ov::element::Type input_type,
        ov::element::Type convert1_out_type,
        ov::element::Type convert2_out_type) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(input_type, input_shape);
        parameter->set_friendly_name("parameter");

        std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::Maximum>(
            parameter,
            std::make_shared<ngraph::opset1::Constant>(input_type, Shape{}, std::vector<float>{0.f}));
        parent->set_friendly_name("maximum");

        parent = convert1_out_type == element::undefined ?
            parent :
            std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convert1_out_type);

        parent = convert2_out_type == element::undefined ?
            parent :
            std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convert2_out_type);

        parent = std::make_shared<ngraph::opset1::Minimum>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(parent->output(0).get_element_type(), Shape{}, std::vector<float>{0.f}));
        parent->set_friendly_name("minimum");

        const auto result = std::make_shared<ngraph::opset1::Result>(parent);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ngraph::ParameterVector{ parameter }, "SnippetsPrecisionPropagation");
    }
};

class RemoveConvertsTestValueItem {
public:
    ov::element::Type convert1_type;
    ov::element::Type convert2_type;
};

class RemoveConvertsTestValue {
public:
    PartialShape input_shape;
    ov::element::Type input_type;
    RemoveConvertsTestValueItem actual;
    RemoveConvertsTestValueItem expected;
};

class RemoveConvertsTests : public ::testing::Test, public testing::WithParamInterface<RemoveConvertsTestValue> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RemoveConvertsTestValue> obj) {
        RemoveConvertsTestValue test_value = obj.param;

        std::ostringstream result;
        result << "IS=" << test_value.input_shape << "_";
        result << "IT=" << test_value.input_type << "_";
        result << "ACT_C1_type=" << test_value.actual.convert1_type << "_";
        result << "ACT_C2_type=" << test_value.actual.convert2_type << "_";
        result << "EXP_C1_type=" << test_value.expected.convert1_type << "_";
        result << "EXP_C2_type=" << test_value.expected.convert2_type << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        RemoveConvertsTestValue test_value = this->GetParam();

        function = RemoveConvertsFunction::get(
            test_value.input_shape,
            test_value.input_type,
            test_value.actual.convert1_type,
            test_value.actual.convert2_type);

        ngraph::pass::Manager manager;
        manager.register_pass<ov::intel_cpu::pass::RemoveConverts>();
        manager.run_passes(function);

        ref_function = RemoveConvertsFunction::get(
            test_value.input_shape,
            test_value.input_type,
            test_value.expected.convert1_type,
            test_value.expected.convert2_type);
    }

    std::shared_ptr<ov::Model> function;
    std::shared_ptr<ov::Model> ref_function;
};

TEST_P(RemoveConvertsTests, RemoveConvertsTests) {
    const auto res = compare_functions(function, ref_function);
    ASSERT_TRUE(res.first) << res.second;
}

namespace RemoveConvertsTestsInstantiation {
std::vector<RemoveConvertsTestValue> test_values = {
    {
        {1, 3, 4, 4},
        element::f32,
        {
            {},
            {}
        },
        {
            {},
            {}
        }
    },
    // I32 => BF16 => FP32
    {
        {1, 3, 4, 4},
        element::i32,
        {
            element::bf16,
            element::f32
        },
        {
            element::bf16,
            element::f32
        }
    },
    // BF16 => BF16 => FP32
    {
        {1, 3, 4, 4},
        element::bf16,
        {
            element::bf16,
            element::f32
        },
        {
            element::bf16,
            element::f32
        }
    },
    // FP32 => I8 => FP32
    {
        {1, 3, 4, 4},
        element::f32,
        {
            element::i8,
            element::f32
        },
        {
            element::i8,
            element::f32
        }
    },
    // FP32 => BF16 => FP32
    {
        {1, 3, 4, 4},
        element::f32,
        {
            element::bf16,
            element::f32
        },
        {
            {},
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, RemoveConvertsTests,
                        ::testing::ValuesIn(test_values),
                        RemoveConvertsTests::getTestCaseName);
} // namespace RemoveConvertsTestsInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov
