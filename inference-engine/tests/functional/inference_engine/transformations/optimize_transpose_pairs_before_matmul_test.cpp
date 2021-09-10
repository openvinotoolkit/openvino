// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/matmul_horizontal_fusing.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

struct Builder {
    std::vector<float> left_values;
    std::vector<float> right_values;
    bool transpose_a;
    bool transpose_b;
};

struct OptimizeTransposePairsTestValues {
    PartialShape input_shape;
    Builder transposes_before;
    Builder transposes_after;
    Shape additional_mul_shape;
};

std::shared_ptr<Function> get(
    const PartialShape& input_shape,
    const Builder& vals,
    const Shape mul_shape) {
    auto input = std::make_shared<opset8::Parameter>(ngraph::element::f32, input_shape);

    const auto transpose_a_const = opset8::Constant::create(element::i64, { vals.left_values.size() }, vals.left_values);
    std::shared_ptr<Node> a_input = std::make_shared<opset8::Transpose>(input, transpose_a_const);
    const auto transpose_b_const = opset8::Constant::create(element::i64, { vals.right_values.size() }, vals.right_values);
    std::shared_ptr<Node> b_input = std::make_shared<opset8::Transpose>(input, transpose_b_const);

    if (!mul_shape.empty()) {
        auto mul_const = opset8::Constant::create(ngraph::element::f32, mul_shape, { 2.f });
        b_input = std::make_shared<opset8::Multiply>(b_input, mul_const);
    }

    const auto matmul = std::make_shared<ngraph::opset8::MatMul>(a_input, b_input, vals.transpose_a, vals.transpose_b);

    return std::make_shared<Function>(OutputVector{ matmul }, ParameterVector{ input });
}

class OptimizeTransposePairs : public ::testing::Test, public testing::WithParamInterface<OptimizeTransposePairsTestValues> {
public:
    void SetUp() override {
        const auto vals = GetParam();

        f = get(vals.input_shape, vals.transposes_before, vals.additional_mul_shape);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::OptimizeTransposePairsBeforeMatMul>();
        manager.run_passes(f);

        f_ref = get(vals.input_shape, vals.transposes_after, vals.additional_mul_shape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<OptimizeTransposePairsTestValues> obj) {
        const auto vals = obj.param;

        std::ostringstream result;
        result << "input_shape_" << vals.input_shape << "before_left_transpose_" << vector_to_string(vals.transposes_before.left_values)
               << "_right_transpose_" << vector_to_string(vals.transposes_before.right_values) << "_matmul_"
               << (vals.transposes_before.transpose_a ? "transpose_a" : "")
               << (vals.transposes_before.transpose_b ? "transpose_b" : "");
        if (!vals.additional_mul_shape.empty()) {
            result << "additional_mul_" << vals.additional_mul_shape;
        }

        result << "after_left_transpose_" << vector_to_string(vals.transposes_after.left_values)
               << "_right_transpose_" << vector_to_string(vals.transposes_after.right_values) << "_matmul_"
               << (vals.transposes_after.transpose_a ? "transpose_a" : "")
               << (vals.transposes_after.transpose_b ? "transpose_b" : "");

        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(OptimizeTransposePairs, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<OptimizeTransposePairsTestValues> test_values = {
    {
        PartialShape{ 1, 2, 128, 64 },
        // actual
        {
            {0, 1, 3, 2}, { 0, 1, 2, 3 }, false, false
        },
        // expected
        {
            {0, 1, 3, 2}, { 0, 1, 3, 2 }, false, true
        }
    },
    {
        PartialShape{ 1, 2, 128, 64 },
        {
            {0, 1, 3, 2}, { 0, 1, 2, 3 }, false, false
        },
        {
            {0, 1, 3, 2}, { 0, 1, 3, 2 }, false, true
        },
        { 1, 2, 1, 1 } // additional mul before matmul
    },
    {
        PartialShape{ 1, 64, 128, 2 },
        {
            {3, 0, 1, 2}, {3, 0, 2, 1}, false, false
        },
        {
            {3, 0, 1, 2}, {3, 0, 1, 2}, false, true
        }
    },
    {
        PartialShape{ 1, 64, 128, 2 },
        {
            {3, 0, 2, 1}, {3, 0, 2, 1}, false, true
        },
        {
            {3, 0, 2, 1}, {3, 0, 2, 1}, false, true
        }
    },
    // additional mul by affected transpose axis
    {
        PartialShape{ 1, 64, 128, 2 },
        {
            {3, 0, 2, 1}, {3, 0, 1, 2}, false, false
        },
        {
            {3, 0, 2, 1}, {3, 0, 2, 1}, false, true
        },
        {1, 1, 1, 1}
    },
    // additional mul by affected transpose axis
    {
        PartialShape{ 1, 64, 128, 2 },
        {
            {3, 0, 2, 1}, {3, 0, 1, 2}, false, false
        },
        {
            {3, 0, 2, 1}, {3, 0, 1, 2}, false, false
        },
        {1, 1, 1, 128}
    },
    // transpose values mismatch after swap last two transpose_right elems
    {
        PartialShape{ 2, 64, 128, 2 },
        {
            {3, 0, 2, 1}, {0, 3, 1, 2}, false, false
        },
        {
            {3, 0, 2, 1}, {0, 3, 1, 2}, false, false
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    OptimizeTransposePairs,
    ::testing::ValuesIn(test_values),
    OptimizeTransposePairs::getTestCaseName);
} // namespace
