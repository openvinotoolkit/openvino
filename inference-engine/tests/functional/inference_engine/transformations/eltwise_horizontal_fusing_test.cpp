// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <transformations/serialize.hpp>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/matmul_horizontal_fusing.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

struct EltwiseBuilder {
    PartialShape shape;
    std::vector<float> values;
    size_t num_of_fused_eltwises;
};

struct EltwiseHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    std::int64_t split_axis;
    std::vector<EltwiseBuilder> eltwises_before;
    std::vector<EltwiseBuilder> eltwises_after;
};

enum EltwiseType {
    SUBTRACT,
    MULTIPLY,
    ADD
};

typedef std::tuple<
    EltwiseHorizontalFusingTestValues,
    EltwiseType
> EltwiseHorizontalFusingParams;

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const EltwiseType eltwise_type,
    const std::int64_t split_axis_value,
    const std::vector<EltwiseBuilder>& eltwise_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };

    OutputVector eltwise_inputs(eltwise_values.size());
    if (eltwise_values.size() == 1 && eltwise_values[0].num_of_fused_eltwises > 0) {
        eltwise_inputs[0] = input->output(0);
    } else {
        const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i32, {}, { split_axis_value });
        const auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, eltwise_inputs.size());
        for (size_t i = 0; i < eltwise_inputs.size(); ++i) {
            eltwise_inputs[i] = split->output(i);
        }
    }
    OutputVector output_nodes;
    for (size_t i = 0; i < eltwise_values.size(); ++i) {
        std::shared_ptr<Node> second_input;
        if (eltwise_values[i].values.empty()) {
            auto input_2 = std::make_shared<opset8::Parameter>(ngraph::element::f32, eltwise_values[i].shape);
            inputs.emplace_back(input_2);
            second_input = input_2;
        } else {
            auto weights_shape = eltwise_values[i].shape.to_shape();
            second_input = opset8::Constant::create(ngraph::element::f32, weights_shape, eltwise_values[i].values);
        }

        std::shared_ptr<ngraph::Node> eltwise;
        if (eltwise_type == EltwiseType::ADD) {
            eltwise = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset8::Add>>(
                element::TypeVector{ element::f32, element::f32 }, element::TypeVector{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(eltwise_inputs[i], element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(second_input, element::f32).get());
        } else if (eltwise_type == EltwiseType::SUBTRACT) {
            eltwise = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset8::Subtract>>(
                element::TypeVector{ element::f32, element::f32 }, element::TypeVector{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(eltwise_inputs[i], element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(second_input, element::f32).get());
        } else if (eltwise_type == EltwiseType::MULTIPLY) {
            eltwise = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset8::Multiply>>(
                element::TypeVector{ element::f32, element::f32 }, element::TypeVector{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(eltwise_inputs[i], element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(second_input, element::f32).get());
        } else {
            throw std::runtime_error("unexpected eltwise type");
        }

        if (eltwise_values[i].num_of_fused_eltwises == 0) {
            output_nodes.emplace_back(eltwise);
        } else {
            const auto split_axis = opset8::Constant::create(element::i32, Shape{}, { split_axis_value });
            const auto split = std::make_shared<opset8::Split>(eltwise, split_axis, eltwise_values[i].num_of_fused_eltwises);
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

class EltwiseHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<EltwiseHorizontalFusingParams> {
public:
    void SetUp() override {
        const auto vals = std::get<0>(GetParam());
        const auto eltwise_type = std::get<1>(GetParam());

        f = get(vals.input_precision, vals.input_shape, eltwise_type, vals.split_axis, vals.eltwises_before);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::AddHorizontalFusing>();
        manager.register_pass<pass::SubtractHorizontalFusing>();
        manager.register_pass<pass::MultiplyHorizontalFusing>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, eltwise_type, vals.split_axis, vals.eltwises_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<EltwiseHorizontalFusingParams> obj) {
        const auto vals = std::get<0>(obj.param);
        const auto eltwise_type = std::get<1>(obj.param);

        std::ostringstream result;
        result << vals.input_shape << "_" << vals.input_precision << "_split_axis_"
               << vals.split_axis << "_" << eltwise_type << "_before_";
        for (const auto& elem : vals.eltwises_before) {
            result << elem.shape;
        }

        result << eltwise_type << "_after_";
        for (const auto& elem : vals.eltwises_after) {
            result << elem.shape << "split_after_has_" << elem.num_of_fused_eltwises << "_splits}_";
        }
        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(EltwiseHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<EltwiseType> eltwise_types = { EltwiseType::ADD, EltwiseType::SUBTRACT, EltwiseType::MULTIPLY };

const std::vector<EltwiseHorizontalFusingTestValues> test_values = {
    {
        element::f32, PartialShape{ 1, 2, 8 }, 2,
        // actual
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        },
        // expected
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ -1, -1, -1 }, 2,
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        },
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 8 }, 2,
        {
            { PartialShape{}, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        },
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 8 }, 2,
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{}, {4} },
        },
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 8 }, 2,
        {
            { PartialShape{}, {2} },
            { PartialShape{}, {4} },
        },
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 8 }, -1,
        // actual
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        },
        // expected
        {
            { PartialShape{ 1, 1, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 4 }, 1,
        {
            { PartialShape{ 1, 2, 1 }, {2} },
            { PartialShape{ 1, 2, 1 }, {4} },
        },
        {
            { PartialShape{ 1, 4, 1 }, {2.f, 2.f, 4.f, 4.f}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 12 }, 2,
        // actual
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
            { PartialShape{ 1, 1, 4 }, {6} },
        },
        // expected
        {
            { PartialShape{ 1, 1, 12 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f}, 3 },
        }
    },
    // ONNX case: 1D constant
    {
        element::f32, PartialShape{ 1, 2, 8 }, 2,
        {
            { PartialShape{ 4 }, {2} },
            { PartialShape{ 4 }, {4} },
        },
        {
            { PartialShape{ 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}, 2 },
        },
    },
    // ONNX case: eltwise axis and split axis mismatch
    {
        element::f32, PartialShape{ 1, 2, 4 }, 1,
        {
            { PartialShape{ 4 }, {2} },
            { PartialShape{ 4 }, {4} },
        },
        {
            { PartialShape{ 4 }, {2} },
            { PartialShape{ 4 }, {4} },
        },
    },
    // ONNX case: impossible to define eltwise axis
    {
        element::f32, PartialShape{ -1, -1, -1 }, 2,
        {
            { PartialShape{ 4 }, {2} },
            { PartialShape{ 4 }, {4} },
        },
        {
            { PartialShape{ 4 }, {2} },
            { PartialShape{ 4 }, {4} },
        },
    },
    // dynamic rank
    {
        element::f32, PartialShape::dynamic(), 1,
        {
            { PartialShape{}, {2} },
            { PartialShape{}, {4} },
        },
        {
            { PartialShape{}, {2} },
            { PartialShape{}, {4} },
        },
    },
    // constant shapes mismatch
    {
        element::f32, PartialShape{ 1, 4, 4 }, 1,
        {
            { PartialShape{ 1, 2, 1 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        },
        {
            { PartialShape{ 1, 2, 1 }, {2} },
            { PartialShape{ 1, 1, 4 }, {4} },
        }
    },
    // non-constant case
    {
        element::f32, PartialShape{ 1, 4, 4 }, 1,
        {
            { PartialShape{ 1, 2, 1 }, {2} },
            { PartialShape{ 1, 2, 1 }, {} },
        },
        {
            { PartialShape{ 1, 2, 1 }, {2} },
            { PartialShape{ 1, 2, 1 }, {} },
        }
    },
    // not all constants match
    {
        element::f32, PartialShape{ 1, 2, 12 }, 2,
        // actual
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 2, 1 }, {4} },
            { PartialShape{ 1, 1, 4 }, {6} },
        },
        // expected
        {
            { PartialShape{ 1, 1, 4 }, {2} },
            { PartialShape{ 1, 2, 1 }, {4} },
            { PartialShape{ 1, 1, 4 }, {6} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    EltwiseHorizontalFusing,
    ::testing::Combine(
        ::testing::ValuesIn(test_values),
        ::testing::ValuesIn(eltwise_types)),
    EltwiseHorizontalFusing::getTestCaseName);
} // namespace
