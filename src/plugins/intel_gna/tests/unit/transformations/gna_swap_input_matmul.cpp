// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/swap_input_matmul_gna.hpp"

namespace testing {

static std::shared_ptr<ngraph::Function> CreateMatMulFunction(const ngraph::Shape& input1_shape,
                                                              const ngraph::Shape& input2_shape,
                                                              const ngraph::Shape& bias_shape,
                                                              bool withBias,
                                                              bool withWeightsFq,
                                                              bool withOutFq,
                                                              bool withAct,
                                                              bool swappedInputs,
                                                              bool needTranspose,
                                                              bool expected = false) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input2_shape);
    std::shared_ptr<ngraph::Node> input = input_params;
    if (input->get_output_shape(0).size() == 2 && needTranspose) {
        auto transpose_order =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>{1, 0});
        input = std::make_shared<ngraph::opset8::Transpose>(input, transpose_order);
    }

    auto constant = ngraph::opset8::Constant::create(ngraph::element::i64, input1_shape, {1});
    std::shared_ptr<ngraph::Node> const_input = constant;
    if (withWeightsFq) {
        auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        const_input = std::make_shared<ngraph::opset8::FakeQuantize>(const_input,
                                                                     input_low,
                                                                     input_high,
                                                                     output_low,
                                                                     output_high,
                                                                     11);
    }
    auto matmul = swappedInputs ? std::make_shared<ngraph::opset8::MatMul>(input, const_input, false, needTranspose)
                                : std::make_shared<ngraph::opset8::MatMul>(const_input, input, needTranspose, false);

    std::shared_ptr<ngraph::Node> final_node = matmul;
    if (withBias) {
        auto shape = bias_shape;
        if ((needTranspose && !expected || !needTranspose && expected) && bias_shape.size() > 1) {
            std::swap(shape[0], shape[1]);
        }
        auto bias = ngraph::opset8::Constant::create(ngraph::element::i64, shape, {1});
        std::shared_ptr<ngraph::Node> bias_node = bias;
        if (expected && bias_shape.size() > 1) {
            auto transpose_order =
                ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>{1, 0});
            bias_node = std::make_shared<ngraph::opset8::Transpose>(bias_node, transpose_order);
        }
        final_node = std::make_shared<ngraph::opset8::Add>(matmul, bias_node);
    }

    if (withOutFq) {
        auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        final_node = std::make_shared<ngraph::opset8::FakeQuantize>(final_node,
                                                                    input_low,
                                                                    input_high,
                                                                    output_low,
                                                                    output_high,
                                                                    11);
    }

    if (withAct) {
        final_node = std::make_shared<ngraph::opset8::Relu>(final_node);
    }

    if (final_node->get_output_shape(0).size() == 2 && needTranspose) {
        auto transpose_order =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>{1, 0});
        final_node = std::make_shared<ngraph::opset8::Transpose>(final_node, transpose_order);
    }

    auto result = std::make_shared<ngraph::opset8::Result>(final_node);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

static void Execute(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::SwapInputMatMulWithTrailingTranspose>();
    m.register_pass<ov::intel_gna::pass::SwapInputMatMulWithAct>();
    m.register_pass<ov::intel_gna::pass::SwapInputMatMulWithFq>();
    m.register_pass<ov::intel_gna::pass::SwapInputMatMulWithBias>();
    m.register_pass<ov::intel_gna::pass::SwapInputMatMul>();
    m.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

typedef std::tuple<std::vector<ngraph::Shape>,  // constant input shape, non-const input shape, bias shape
                   bool,                        // with bias
                   bool,                        // with weights FakeQuantize
                   bool,                        // with output FakeQuantize
                   bool,                        // with activation after MatMul
                   bool                         // with transposes
                   >
    SwapInputMatmulParams;

static std::string getTestCaseName(testing::TestParamInfo<SwapInputMatmulParams> obj) {
    std::vector<ngraph::Shape> shapes;
    bool withBias, withWeightsFq, withOutFq, withAct, withTransposes;
    std::tie(shapes, withBias, withWeightsFq, withOutFq, withAct, withTransposes) = obj.param;

    std::ostringstream result;
    result << "IS1=" << shapes[0] << "_";
    result << "IS2=" << shapes[1] << "_";
    result << "BS=" << shapes[2] << "_";
    result << "bias=" << withBias << "_";
    result << "wFQ=" << withWeightsFq << "_";
    result << "oFQ=" << withOutFq << "_";
    result << "act=" << withAct << "_";
    result << "transposes=" << withTransposes;
    return result.str();
}

enum class MatmulInputType { FirstInputConstant, SecondInputConstant };  // enum class MatmulInputType

static void transposeInputShapes(std::vector<ngraph::Shape>& shapes) {
    if (shapes[0].size() > 1) {
        std::swap(shapes[0][0], shapes[0][1]);
    }
    if (shapes[1].size() > 1) {
        std::swap(shapes[1][0], shapes[1][1]);
    }
    if (shapes[2].size() > 1) {
        std::swap(shapes[2][0], shapes[2][1]);
    }
}

template <MatmulInputType E>
class SwapInputMatmul : public ov::test::TestsCommon, public ::testing::WithParamInterface<SwapInputMatmulParams> {
public:
    void SetUp() override {
        std::vector<ngraph::Shape> shapes;
        bool withBias, withWeightsFq, withOutFq, withAct, withTransposes;
        std::tie(shapes, withBias, withWeightsFq, withOutFq, withAct, withTransposes) = this->GetParam();

        bool swap_inputs = false;
        switch (E) {
        case MatmulInputType::FirstInputConstant:
            break;
        case MatmulInputType::SecondInputConstant:
            swap_inputs = true;
            break;
        }

        if (withTransposes) {
            transposeInputShapes(shapes);
        }
        function = CreateMatMulFunction(shapes[0],
                                        shapes[1],
                                        shapes[2],
                                        withBias,
                                        withWeightsFq,
                                        withOutFq,
                                        withAct,
                                        swap_inputs,
                                        withTransposes);
        reference_function = CreateMatMulFunction(shapes[0],
                                                  shapes[1],
                                                  shapes[2],
                                                  withBias,
                                                  withWeightsFq,
                                                  withOutFq,
                                                  withAct,
                                                  !swap_inputs,
                                                  !withTransposes,
                                                  true);
    }

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

template <MatmulInputType E>
class SwapInputMatmulNotApplied : public ov::test::TestsCommon,
                                  public ::testing::WithParamInterface<SwapInputMatmulParams> {
public:
    void SetUp() override {
        std::vector<ngraph::Shape> shapes;
        bool withBias, withWeightsFq, withOutFq, withAct, withTransposes;
        std::tie(shapes, withBias, withWeightsFq, withOutFq, withAct, withTransposes) = this->GetParam();

        bool swap_inputs = false;
        switch (E) {
        case MatmulInputType::FirstInputConstant:
            break;
        case MatmulInputType::SecondInputConstant:
            swap_inputs = true;
            break;
        }

        if (withTransposes) {
            transposeInputShapes(shapes);
        }
        function = CreateMatMulFunction(shapes[0],
                                        shapes[1],
                                        shapes[2],
                                        withBias,
                                        withWeightsFq,
                                        withOutFq,
                                        withAct,
                                        swap_inputs,
                                        withTransposes);
        reference_function = ngraph::clone_function(*function);
    }

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

using SwapInputMatmulWithFirstInputConstant = SwapInputMatmul<MatmulInputType::FirstInputConstant>;
using SwapInputMatmulWithSecondInputConstant = SwapInputMatmul<MatmulInputType::SecondInputConstant>;
using SwapInputMatmulWithFirstInputConstantNotApplied = SwapInputMatmulNotApplied<MatmulInputType::FirstInputConstant>;
using SwapInputMatmulWithSecondInputConstantNotApplied =
    SwapInputMatmulNotApplied<MatmulInputType::SecondInputConstant>;

TEST_P(SwapInputMatmulWithFirstInputConstant, CompareFunctions) {
    Execute(function, reference_function);
}

TEST_P(SwapInputMatmulWithFirstInputConstantNotApplied, CompareFunctions) {
    Execute(function, reference_function);
}

TEST_P(SwapInputMatmulWithSecondInputConstant, CompareFunctions) {
    Execute(function, reference_function);
}

TEST_P(SwapInputMatmulWithSecondInputConstantNotApplied, CompareFunctions) {
    Execute(function, reference_function);
}

const std::vector<std::vector<ngraph::Shape>> input_shapes_for_matmul_with_first_constant_applied = {
    {{16, 8}, {8, 8}, {16, 8}},
    {{16, 8}, {8, 8}, {1}},
};

const std::vector<std::vector<ngraph::Shape>> input_shapes_for_matmul_with_first_constant_not_applied = {
    {{1, 8}, {8, 8}, {1, 8}},
    {{8}, {8, 8}, {8}}};

const std::vector<std::vector<ngraph::Shape>> input_shapes_for_matmul_with_second_constant_applied = {
    {{64, 6}, {100, 64}, {100, 6}},
    {{64, 6}, {100, 64}, {1}},
};

const std::vector<std::vector<ngraph::Shape>> input_shapes_for_matmul_with_second_constant_not_applied = {
    {{64, 16}, {100, 64}, {100, 16}},
    {{64, 6}, {8, 64}, {8, 6}},
    {{8, 1}, {8, 8}, {8, 1}},
    {{8}, {8, 8}, {8}}};

INSTANTIATE_TEST_SUITE_P(smoke_swap_input_matmul,
                         SwapInputMatmulWithFirstInputConstant,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_for_matmul_with_first_constant_applied),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true})),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_swap_input_matmul,
    SwapInputMatmulWithFirstInputConstantNotApplied,
    ::testing::Combine(::testing::ValuesIn(input_shapes_for_matmul_with_first_constant_not_applied),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true})),
    getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_swap_input_matmul,
                         SwapInputMatmulWithSecondInputConstant,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_for_matmul_with_second_constant_applied),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),
                                            ::testing::ValuesIn(std::vector<bool>{false, true})),
                         getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_swap_input_matmul,
    SwapInputMatmulWithSecondInputConstantNotApplied,
    ::testing::Combine(::testing::ValuesIn(input_shapes_for_matmul_with_second_constant_not_applied),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<bool>{false, true})),
    getTestCaseName);

}  // namespace testing
