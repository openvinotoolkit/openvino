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
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

struct MatMulBuilder {
    struct WeightsPath {
        element::Type precision;
        PartialShape shape;
        std::vector<float> weights_values;
        std::vector<float> sub_deq_values;
        std::vector<float> mul_deq_values;
    };

    WeightsPath weights;
    bool transpose_a;
    bool transpose_b;
    size_t num_of_fused_matmuls;
};

struct MatMulHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    std::vector<MatMulBuilder> matmuls_before;
    std::vector<MatMulBuilder> matmuls_after;
    bool add_additional_consumer;
};

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const std::vector<MatMulBuilder>& matmul_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };
    OutputVector matmul_inputs(matmul_values.size());

    auto relu = std::make_shared<opset8::Relu>(input);
    for (auto& elem : matmul_inputs) {
        elem = relu->output(0);
    }

    OutputVector output_nodes;
    for (size_t i = 0; i < matmul_values.size(); ++i) {
        std::shared_ptr<Node> second_input;
        if (matmul_values[i].weights.weights_values.empty()) {
            auto input_2 = std::make_shared<opset8::Parameter>(matmul_values[i].weights.precision, matmul_values[i].weights.shape);
            inputs.emplace_back(input_2);
            second_input = input_2;
        } else {
            auto weights_shape = matmul_values[i].weights.shape.to_shape();
            second_input = opset8::Constant::create(matmul_values[i].weights.precision, weights_shape, matmul_values[i].weights.weights_values);
            if (matmul_values[i].weights.precision != element::f32 &&
                (!matmul_values[i].weights.sub_deq_values.empty() || !matmul_values[i].weights.mul_deq_values.empty())) {
                second_input = std::make_shared<opset8::Convert>(second_input, element::f32);

                Shape deq_const_shape(2, 1ul);
                size_t out_channel_idx = matmul_values[i].transpose_b ? weights_shape.size() - 2 : weights_shape.size() - 1;
                deq_const_shape[out_channel_idx] = weights_shape[out_channel_idx];

                if (!matmul_values[i].weights.sub_deq_values.empty()) {
                    auto sub_const = opset8::Constant::create(element::f32, deq_const_shape, matmul_values[i].weights.sub_deq_values);
                    second_input = std::make_shared<opset8::Subtract>(second_input, sub_const);
                }

                if (!matmul_values[i].weights.mul_deq_values.empty()) {
                    auto mul_const = opset8::Constant::create(element::f32, deq_const_shape, matmul_values[i].weights.mul_deq_values);
                    second_input = std::make_shared<opset8::Multiply>(second_input, mul_const);
                }
            }
        }

        const auto matmul = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::MatMul>>(
            std::vector<element::Type>{ element::f32, element::f32 }, std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(matmul_inputs[i], element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(second_input, element::f32).get(),
            matmul_values[i].transpose_a,
            matmul_values[i].transpose_b);

        if (matmul_values[i].num_of_fused_matmuls == 0) {
            output_nodes.emplace_back(matmul);
        } else {
            const auto matmul_out_rank = matmul->get_output_partial_shape(0).size();
            const size_t split_axis_value = matmul_out_rank - 1;

            const auto split_axis = opset8::Constant::create(element::i64, Shape{}, { split_axis_value });
            const auto split = std::make_shared<opset8::Split>(matmul, split_axis, matmul_values[i].num_of_fused_matmuls);
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

class MatMulHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<MatMulHorizontalFusingTestValues> {
public:
    void SetUp() override {
        const auto vals = GetParam();

        f = get(vals.input_precision, vals.input_shape, vals.matmuls_before);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::MatMulHorizontalFusing>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, vals.matmuls_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulHorizontalFusingTestValues> obj) {
        const auto vals = obj.param;

        std::ostringstream result;
        result << vals.input_shape << "_" << vals.input_precision << "_matmuls_before_";

        for (const auto& elem : vals.matmuls_before) {
            result << "{weights_" << elem.weights.precision << "_" << elem.weights.shape << "_sub_"
                   << vector_to_string(elem.weights.sub_deq_values) << "_mul_" << vector_to_string(elem.weights.mul_deq_values)
                   << "}_" << (elem.transpose_a ? "transpose_a" : "") << (elem.transpose_b ? "transpose_b" : "");
        }

        result << "matmuls_after_";
        for (const auto& elem : vals.matmuls_after) {
            result << "{weights_" << elem.weights.precision << "_" << elem.weights.shape << "_sub_"
                   << vector_to_string(elem.weights.sub_deq_values) << "_mul_" << vector_to_string(elem.weights.mul_deq_values)
                   << "split_after_has_" << elem.num_of_fused_matmuls << "_splits}_"
                   << "}_" << (elem.transpose_a ? "transpose_a" : "") << (elem.transpose_b ? "transpose_b" : "");
        }

        result << (vals.add_additional_consumer ? "additional_consumer" : "");
        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(MatMulHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<MatMulHorizontalFusingTestValues> test_values = {
    {
        element::f32, PartialShape{ 1, 2, 4 },
        // actual
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        // expected
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, false, true },
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, false, true },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                false, true, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 2 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, true, true },
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, true, true },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                true, true, 2
            }
        }
    },
    {
        element::u8, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {2}, {0.f, 0.f, 0.f, 0.f}, {0.1f, 0.1f, 0.1f, 0.1f} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4}, {0.f, 0.f, 0.f, 0.f}, {0.2f, 0.2f, 0.2f, 0.2f} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f},
                                                                               {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                                                                               {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f}},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ -1, -1, -1 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {6} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 12 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f}},
                false, false, 3
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {6} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 12 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f,
                                                                                  2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f, 6.f, 6.f, 6.f, 6.f}},
                false, false, 3
            }
        }, true // add consumer (not matmul)
    },
    // fused only 2 matmuls
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6} }}
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                false, false, 2
            },
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6}} }
        }
    },
    {
        element::u8, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {2}, {}, {} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4}, {}, {3.f} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {2}, {}, {} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4}, {}, {3.f} }},
        }
    },
    // dynamic rank
    {
        element::f32, PartialShape::dynamic(),
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
        }
    },
    // different weights precision
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        }
    },
    // different transpose flags
    {
        element::f32, PartialShape{ 1, 4, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, true, false},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, true, false},
        }
    },
    // matmul with two activations
    {
        element::f32, PartialShape{ 1, 2, 4 },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {} }},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {} }},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    MatMulHorizontalFusing,
    ::testing::ValuesIn(test_values),
    MatMulHorizontalFusing::getTestCaseName);
} // namespace
