// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/pass/visualize_tree.hpp>

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
        std::vector<float> values;
    };

    struct BiasPath {
        Shape shape;
        std::vector<float> values;
    };

    MatMulBuilder(WeightsPath weights, BiasPath bias = {}, bool transpose_a = false, bool transpose_b = false, size_t splits_after = 0) :
        weights(weights), bias(bias), transpose_a(transpose_a), transpose_b(transpose_b), num_splits_after(splits_after) {}

    WeightsPath weights;
    BiasPath bias;
    bool transpose_a;
    bool transpose_b;
    size_t num_splits_after;
};

enum AdditionalOp {
    CLAMP,
    NONE,
};

struct MatMulHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    AdditionalOp additional_consumer;
    std::vector<MatMulBuilder> matmuls_before;
    std::vector<MatMulBuilder> matmuls_after;
};

typedef std::tuple <
    MatMulHorizontalFusingTestValues,
    bool, // add fq
    bool, // add reshape
    bool, // add transpose
    bool, // add sub
    bool  // add mul
> MatMulHorizontalFusingParams;

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const AdditionalOp additional_op,
    const std::vector<MatMulBuilder>& matmul_values,
    const bool add_fq,
    const bool add_reshape,
    const bool add_transpose,
    const bool add_sub,
    const bool add_mul) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };
    auto relu = std::make_shared<opset8::Relu>(input);

    NodeVector results;
    if (AdditionalOp::CLAMP) {
        auto clamp = std::make_shared<opset8::Clamp>(relu, 0.0, 6.0);
        results.emplace_back(clamp);
    }

    OutputVector output_nodes;
    for (const auto& matmul_val : matmul_values) {
        std::shared_ptr<Node> weights;
        if (matmul_val.weights.values.empty()) {
            auto input_2 = std::make_shared<opset8::Parameter>(matmul_val.weights.precision, matmul_val.weights.shape);
            inputs.emplace_back(input_2);
            weights = input_2;
        } else {
            auto weights_shape = matmul_val.weights.shape.to_shape();
            weights = opset8::Constant::create(matmul_val.weights.precision, weights_shape, matmul_val.weights.values);
            if (matmul_val.weights.precision != element::f32) {
                weights = std::make_shared<opset8::Convert>(weights, element::f32);

                Shape deq_const_shape(2, 1ul);
                if (matmul_val.weights.values.size() > 1) {
                    size_t out_channel_idx = matmul_val.transpose_b ? weights_shape.size() - 2 : weights_shape.size() - 1;
                    deq_const_shape[out_channel_idx] = weights_shape[out_channel_idx];
                }

                auto sub_const = opset8::Constant::create(element::f32, deq_const_shape, { 0.0001f });
                weights = std::make_shared<opset8::Subtract>(weights, sub_const);

                auto mul_const = opset8::Constant::create(element::f32, deq_const_shape, { 0.56f });
                weights = std::make_shared<opset8::Multiply>(weights, mul_const);
            }
        }

        std::shared_ptr<Node> last_node = std::make_shared<opset8::MatMul>(relu, weights, matmul_val.transpose_a, matmul_val.transpose_b);
        auto matmul_out_pshape = last_node->get_output_partial_shape(0);
        auto matmul_out_rank = matmul_out_pshape.rank();

        if (matmul_out_rank.is_static()) {
            ngraph::Shape constant_shape(matmul_out_rank.is_static());
            constant_shape.back() = matmul_out_pshape[matmul_out_rank.get_length() - 1].get_length();

            if (add_sub) {
                auto sub_const = opset8::Constant::create(element::f32, constant_shape, { 3600 });
                auto sub = std::make_shared<opset8::Subtract>(last_node, sub_const);
                last_node = sub;
            }

            if (add_mul) {
                auto mul_const = opset8::Constant::create(element::f32, constant_shape, { 3600 });
                auto mul = std::make_shared<opset8::Multiply>(last_node, mul_const);
                last_node = mul;
            }
        }

        if (!matmul_val.bias.values.empty()) {
            auto bias_const = opset8::Constant::create(element::f32, matmul_val.bias.shape, matmul_val.bias.values);
            auto bias = std::make_shared<opset8::Add>(last_node, bias_const);
            last_node = bias;
        }

        if (add_fq && matmul_out_rank.is_static()) {
            ngraph::Shape intervals_shape(matmul_out_rank.get_length(), 1);
            intervals_shape[intervals_shape.size() - 1] = matmul_out_pshape[intervals_shape.size() - 1].get_length();
            const auto il = ngraph::opset8::Constant::create(element::f32, intervals_shape, { -3600 });
            const auto ih = ngraph::opset8::Constant::create(element::f32, intervals_shape, { 3600 });
            const auto ol = ngraph::opset8::Constant::create(element::f32, ngraph::Shape{}, { -128 });
            const auto oh = ngraph::opset8::Constant::create(element::f32, ngraph::Shape{}, { 127 });
            const auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(last_node, il, ih, ol, oh, 256);
            last_node = fq;
        }

        if (add_reshape) {
            std::vector<std::int64_t> reshape_values = { 1, 2, 2, 2 };
            if (matmul_val.num_splits_after > 0) {
                reshape_values[reshape_values.size() - 2] *= matmul_val.num_splits_after;
            }
            const auto reshape_const = opset8::Constant::create(element::i64, Shape{ reshape_values.size() }, reshape_values);
            const auto reshape = std::make_shared<opset8::Reshape>(last_node, reshape_const, true);
            last_node = reshape;
        }

        if (add_transpose) {
            std::vector<std::int64_t> transpose_values = { 0, 2, 1 };
            auto prev_node_rank = last_node->get_output_partial_shape(0).rank();
            if (prev_node_rank.is_static() && prev_node_rank.get_length() == 4) {
                transpose_values.emplace_back(3);
            }
            const auto transpose_const = opset8::Constant::create(element::i64, Shape{ transpose_values.size() }, transpose_values);
            const auto transpose = std::make_shared<opset8::Transpose>(last_node, transpose_const);
            last_node = transpose;
        }

        if (matmul_val.num_splits_after == 0) {
            output_nodes.emplace_back(last_node);
        } else {
            auto split_axis = opset8::Constant::create(element::i64, Shape{}, { add_transpose ? 1 : 2});
            last_node = std::make_shared<opset8::Split>(last_node, split_axis, matmul_val.num_splits_after);
            auto outputs = last_node->outputs();
            for (const auto& out : outputs) {
                output_nodes.emplace_back(out);
            }
        }
    }

    for (const auto node : output_nodes) {
        auto relu = std::make_shared<opset8::Relu>(node);
        results.emplace_back(relu);
    }

    return std::make_shared<Function>(results, inputs);
}

class MatMulHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<MatMulHorizontalFusingParams> {
public:
    void SetUp() override {
        const auto vals = std::get<0>(GetParam());
        const bool add_fq = std::get<1>(GetParam());
        const bool add_reshape = std::get<2>(GetParam());
        const bool add_transpose = std::get<3>(GetParam());
        const bool add_sub = std::get<4>(GetParam());
        const bool add_mul = std::get<5>(GetParam());

        f = get(vals.input_precision, vals.input_shape, vals.additional_consumer, vals.matmuls_before, add_fq, add_reshape, add_transpose, add_sub, add_mul);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::MatMulHorizontalFusion>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, vals.additional_consumer, vals.matmuls_after, add_fq, add_reshape, add_transpose, add_sub, add_mul);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MatMulHorizontalFusingParams> obj) {
        const auto testValues = std::get<0>(obj.param);
        const bool add_fq = std::get<1>(obj.param);
        const bool add_reshape = std::get<2>(obj.param);
        const bool add_transpose = std::get<3>(obj.param);
        const bool add_sub = std::get<4>(obj.param);
        const bool add_mul = std::get<5>(obj.param);

        std::ostringstream result;
        result << testValues.input_shape << "_" << testValues.input_precision << "_matmuls_before_"
               << (testValues.additional_consumer == AdditionalOp::CLAMP ? "additional_op_" : "");
        for (const auto& elem : testValues.matmuls_before) {
            result << "{weights_" << elem.weights.precision << elem.weights.shape;
            if (!elem.bias.values.empty()) {
                result << "_bias_" << elem.bias.shape;
            }
            result << (elem.transpose_a ? "transpose_a_" : "") << (elem.transpose_b ? "transpose_b_" : "") << "}_";
        }
        result << "matmuls_after_";
        for (const auto& elem : testValues.matmuls_after) {
            result << "{weights_" << elem.weights.precision << elem.weights.shape;
            if (!elem.bias.values.empty()) {
                result << "_bias_" << elem.bias.shape;
            }
            result << (elem.transpose_a ? "transpose_a_" : "") << (elem.transpose_b ? "transpose_b_" : "") << "}_";
            if (elem.num_splits_after > 0) {
                result << "split_into_" << elem.num_splits_after << "_outputs";
            }
            result << "}_";
        }


        result << (add_fq ? "fq_" : "") << (add_reshape ? "reshape_" : "") << (add_transpose ? "transpose" : "")
               << (add_sub ? "sub_" : "") << (add_mul ? "mul_" : "");
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

namespace positive_cases {
const std::vector<MatMulHorizontalFusingTestValues> test_values {
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
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
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::CLAMP,
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
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, false, true},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, true, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 2 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, true, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, true, true},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 8, 4 }, {2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, true, true, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {4} }},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 8 }, {2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4,
                                                                                 2, 2, 2, 2, 4, 4, 4, 4}},
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape::dynamic(3), AdditionalOp::NONE,
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
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} },
                MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, { 15.f} }
            },
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} },
                MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, { 30.f} }
            },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{Shape{ 1, 1, 8 }, { 15.f, 15.f, 15.f, 15.f, 30.f, 30.f, 30.f, 30.f }},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} },
                MatMulBuilder::BiasPath{ Shape{ 4 }, { 15.f} }
            },
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} },
                MatMulBuilder::BiasPath{ Shape{ 4 }, { 30.f} }
            },
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{Shape{ 8 }, { 15.f, 15.f, 15.f, 15.f, 30.f, 30.f, 30.f, 30.f }},
                false, false, 2
            }
        }
    },
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
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
                MatMulBuilder::BiasPath{}, false, false, 3
            }
        }
    },
    // fused only 2 matmuls
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }},
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6} }, MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, {15.f}}},
        },
        {
            {
                MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 8 }, {2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f,
                                                                                 2.f, 2.f, 2.f, 2.f, 4.f, 4.f, 4.f, 4.f}},
                MatMulBuilder::BiasPath{}, false, false, 2
            },
            { MatMulBuilder::WeightsPath{ element::i8, PartialShape{ 4, 4 }, {6} }, MatMulBuilder::BiasPath{ Shape{ 1, 1, 4 }, {15.f}}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    MatMulHorizontalFusing,
    ::testing::Combine(
        ::testing::ValuesIn(test_values),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false)),
    MatMulHorizontalFusing::getTestCaseName);
} // namespace positive_cases

namespace positive_cases_with_additional_layers {
const std::vector<MatMulHorizontalFusingTestValues> test_values {
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
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
                MatMulBuilder::BiasPath{}, false, false, 2
            }
        }
    }
};

const std::vector<bool> add_fq = { true, false };

const std::vector<bool> add_reshape = { true, false };

const std::vector<bool> add_transpose = { true, false };

const std::vector<bool> add_sub = { true, false };

const std::vector<bool> add_mul = { true, false };

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    MatMulHorizontalFusing,
    ::testing::Combine(
        ::testing::ValuesIn(test_values),
        ::testing::ValuesIn(add_fq),
        ::testing::ValuesIn(add_reshape),
        ::testing::ValuesIn(add_transpose),
        ::testing::ValuesIn(add_sub),
        ::testing::ValuesIn(add_mul)),
    MatMulHorizontalFusing::getTestCaseName);
} // namespace positive_cases_with_additional_layers

namespace negative_cases {
const std::vector<MatMulHorizontalFusingTestValues> test_values{
    // dynamic rank
    {
        element::f32, PartialShape::dynamic(), AdditionalOp::NONE,
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
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
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
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, {}, true, false},
        },
        {
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {2} }, {}, {}, false, true},
            { MatMulBuilder::WeightsPath{ element::f32, PartialShape{ 4, 4 }, {4} }, {}, {}, true, false},
        }
    },
    // matmul with two activations
    {
        element::f32, PartialShape{ 1, 2, 4 }, AdditionalOp::NONE,
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
    ::testing::Combine(
        ::testing::ValuesIn(test_values),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false)),
    MatMulHorizontalFusing::getTestCaseName);
} // namespace negative_cases
} // namespace
