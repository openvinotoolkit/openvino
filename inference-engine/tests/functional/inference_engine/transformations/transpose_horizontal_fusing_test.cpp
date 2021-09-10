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

struct TransposeBuilder {
    std::vector<float> values;
    size_t num_of_fused_transposes;
};

struct TransposeHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    std::int64_t split_axis_before;
    std::vector<TransposeBuilder> transposes_before;
    std::int64_t split_axis_after;
    std::vector<TransposeBuilder> transposes_after;
};

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const std::int64_t split_axis_value,
    const std::vector<TransposeBuilder>& transpose_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };

    OutputVector transpose_inputs(transpose_values.size());
    if (transpose_values.size() == 1 && transpose_values[0].num_of_fused_transposes > 0) {
        transpose_inputs[0] = input->output(0);
    } else {
        const auto split_axis_before = ngraph::opset8::Constant::create(ngraph::element::i32, {}, { split_axis_value });
        const auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis_before, transpose_inputs.size());
        for (size_t i = 0; i < transpose_inputs.size(); ++i) {
            transpose_inputs[i] = split->output(i);
        }
    }

    OutputVector output_nodes;
    for (size_t i = 0; i < transpose_values.size(); ++i) {
        std::shared_ptr<Node> second_input;
        const auto reshape_const = opset8::Constant::create(ngraph::element::i64, { transpose_values[i].values.size() }, transpose_values[i].values);
        const auto transpose = std::make_shared<ngraph::opset8::Transpose>(transpose_inputs[i], reshape_const);

        if (transpose_values[i].num_of_fused_transposes == 0) {
            output_nodes.emplace_back(transpose);
        } else {
            const auto split_axis_before = opset8::Constant::create(element::i32, Shape{}, { split_axis_value });
            const auto split = std::make_shared<opset8::Split>(transpose, split_axis_before, transpose_values[i].num_of_fused_transposes);
            const auto outputs = split->outputs();
            for (const auto& out : outputs) {
                output_nodes.emplace_back(out);
            }
        }
    }

    ResultVector results;
    for (const auto& node : output_nodes) {
        const auto result_node = std::make_shared<ngraph::opset8::Relu>(node);
        results.emplace_back(std::make_shared<ngraph::opset8::Result>(result_node));
    }

    return std::make_shared<Function>(results, inputs);
}

class TransposeHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<TransposeHorizontalFusingTestValues> {
public:
    void SetUp() override {
        const auto vals = GetParam();

        f = get(vals.input_precision, vals.input_shape, vals.split_axis_before, vals.transposes_before);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::TransposeHorizontalFusing>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, vals.split_axis_after, vals.transposes_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<TransposeHorizontalFusingTestValues> obj) {
        const auto vals = obj.param;

        std::ostringstream result;
        result << vals.input_shape << "_" << vals.input_precision << "_split_axis_before_"
               << vals.split_axis_before << "_transposes_before_";
        for (const auto& elem : vals.transposes_before) {
            result << vector_to_string(elem.values) << "_";
        }

        result << "transposes_after_";
        for (const auto& elem : vals.transposes_after) {
            result << vector_to_string(elem.values) << "_" << elem.num_of_fused_transposes << "_splits}_";
        }
        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(TransposeHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<TransposeHorizontalFusingTestValues> test_values = {
    {
        element::f32, PartialShape{ 1, 128, 66, 2 }, 2,
        // actual
        {
            {{0, 1, 3, 2}},
            {{0, 1, 3, 2}},
            {{0, 1, 3, 2}},
        },
        // expected
        3,
        {
            {{0, 1, 3, 2}, 3},
        }
    },
    {
        element::f32, PartialShape{ 1, 128, 64, 2 }, 2,
        {
            {{0, 1, 3, 2}},
            {{0, 1, 3, 2}},
        },
        3,
        {
            {{0, 1, 3, 2}, 2},
        }
    },
    {
        element::f32, PartialShape{ 1, 128, 64, 2 }, 2,
        // actual
        {
            {{3, 0, 1, 2}},
            {{3, 0, 1, 2}},
        },
        // expected
        3,
        {
            {{3, 0, 1, 2}, 2},
        }
    },
    {
        element::f32, PartialShape{ -1, -1, -1, -1 }, 2,
        // actual
        {
            {{3, 0, 1, 2}},
            {{3, 0, 1, 2}},
        },
        // expected
        3,
        {
            {{3, 0, 1, 2}, 2},
        }
    },
    {
        element::f32, PartialShape::dynamic(), 2,
        // actual
        {
            {{3, 0, 1, 2}},
            {{3, 0, 1, 2}},
        },
        // expected
        3,
        {
            {{3, 0, 1, 2}, 2},
        }
    },
    // different transpose values
    {
        element::f32, PartialShape{ 1, 128, 66, 2 }, 2,
        // actual
        {
            {{0, 1, 3, 2}},
            {{0, 1, 3, 2}},
            {{0, 3, 1, 2}},
        },
        // expected
        2,
        {
            {{0, 1, 3, 2}},
            {{0, 1, 3, 2}},
            {{0, 3, 1, 2}},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    TransposeHorizontalFusing,
    ::testing::ValuesIn(test_values),
    TransposeHorizontalFusing::getTestCaseName);
} // namespace
