// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph_functions/builders.hpp>
#include <transformations/init_node_info.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>
#include <transformations/utils/utils.hpp>
#include <legacy/ngraph_ops/eltwise.hpp>

#include "ops/identity.hpp"
#include "transformations/insert_identity_layer.hpp"
#include "transformations/rt_info/gna_precision_change_flag.hpp"

namespace testing {

class InsertIdentityLayerTest: public CommonTestUtils::TestsCommon {
public:
    virtual void Validate();
    virtual void Run();
public:
    std::shared_ptr<ngraph::Function> m_func, m_ref_func;
    ngraph::Shape m_input_shape{10};
    bool m_low_precision = false;
};

void InsertIdentityLayerTest::Validate() {
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::MarkIdentityCandidates>(m_low_precision);
    m.register_pass<ov::intel_gna::pass::InsertIdentity>();
    m.register_pass<ov::intel_gna::pass::BreakFusingOfOutputLayers>();
    m.run_passes(m_func);
    ASSERT_NO_THROW(check_rt_info(m_func));

    auto result = compare_functions(m_func, m_ref_func);
    ASSERT_TRUE(result.first);

    // Cleanup rt info and check
    m.register_pass<ov::intel_gna::pass::IdentityCandidatesCleanup>();
    m.run_passes(m_func);
    for (auto& node : m_func->get_ordered_ops()) {
        for (auto& input : node->inputs()) {
            const ov::RTMap& rt_info = input.get_rt_info();
            ASSERT_EQ(rt_info.count(ov::intel_gna::rt_info::GNAPrecisionChangeFlag::get_type_info_static()), 0);
        }
    }
}

void InsertIdentityLayerTest::Run() {
    SetUp();
    Validate();
}

/******************************************************* Concat layer tests *******************************************************/

typedef std::tuple<
        size_t,    // Concat axis
        size_t     // input number
> InsertIdentityConcatTestParams;

class InsertIdentityLayerConcatTest: public InsertIdentityLayerTest,
                                     public ::testing::WithParamInterface<InsertIdentityConcatTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertIdentityConcatTestParams>& obj) {
        size_t axis, inputs_num;
        std::tie(axis, inputs_num) = obj.param;

        std::ostringstream result;
        result << "inputsNum=" << inputs_num << "_";
        result << "axis=" << axis;

        return result.str();
    }
    void SetUp() override {
        size_t axis, inputs_num;
        std::tie(axis, inputs_num) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            ngraph::OutputVector concat_inputs = {add};
            for (size_t i = 1; i < inputs_num; ++i) {
                auto const_mul = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {i});
                auto mul = std::make_shared<ngraph::opset9::Multiply>(add, const_mul);
                concat_inputs.push_back(mul);
            }
            auto concat = std::make_shared<ngraph::opset9::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset9::Result>(concat);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params});
        }

        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            auto identity = std::make_shared<ov::intel_gna::op::Identity>(add);
            ngraph::OutputVector concat_inputs = {identity};
            for (size_t i = 1; i < inputs_num; ++i) {
                auto const_mul = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {i});
                auto mul = std::make_shared<ngraph::opset9::Multiply>(identity, const_mul);
                auto identity_mul = std::make_shared<ov::intel_gna::op::Identity>(mul);
                concat_inputs.push_back(identity_mul);
            }
            auto concat = std::make_shared<ngraph::opset9::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset9::Result>(concat);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params});
        }
    }
};

const size_t axis = 0;
const std::vector<size_t> inputCounts = {1, 8};

TEST_P(InsertIdentityLayerConcatTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertIdentityLayerConcatTest,
                         ::testing::Combine(
                                ::testing::Values(axis),
                                ::testing::ValuesIn(inputCounts)),
                         InsertIdentityLayerConcatTest::getTestCaseName);

/******************************************************* Split layer tests *******************************************************/

class InsertIdentityLayerSplitTest: public InsertIdentityLayerTest {
public:
    void SetUp() override {
        InsertIdentityLayerTest::SetUp();
        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            auto axis_const = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
            auto split = std::make_shared<ngraph::opset9::Split>(add, axis_const, 2);
            auto result1 = std::make_shared<ngraph::opset9::Result>(split->output(0));
            auto const_reshape = ngraph::opset9::Constant::create(ngraph::element::i64, {2}, {1, 5});
            auto reshape = std::make_shared<ngraph::opset9::Reshape>(split->output(1), const_reshape, false);
            auto const_mul = ngraph::opset9::Constant::create(ngraph::element::f32, {1, 5}, {1});
            auto mul = std::make_shared<ngraph::opset9::Multiply>(reshape, const_mul);
            auto result2 = std::make_shared<ngraph::opset9::Result>(mul);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                        ngraph::ParameterVector{params});
        }

        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            auto identity = std::make_shared<ov::intel_gna::op::Identity>(add);
            auto axis_const = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
            auto split = std::make_shared<ngraph::opset9::Split>(identity, axis_const, 2);
            auto result1 = std::make_shared<ngraph::opset9::Result>(split->output(0));
            auto const_reshape = ngraph::opset9::Constant::create(ngraph::element::i64, {2}, {1, 5});
            auto reshape = std::make_shared<ngraph::opset9::Reshape>(split->output(1), const_reshape, false);
            auto const_mul = ngraph::opset9::Constant::create(ngraph::element::f32, {1, 5}, {1});
            auto mul = std::make_shared<ngraph::opset9::Multiply>(reshape, const_mul);
            auto result2 = std::make_shared<ngraph::opset9::Result>(mul);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                            ngraph::ParameterVector{params});
        }
    }
};

TEST_F(InsertIdentityLayerSplitTest, CompareWithRefs) {
    Run();
}

/******************************************************* Eltwise layer tests *******************************************************/

typedef std::tuple<
        ELTWISE_TYPE,   // eltwise type
        bool,           // use low precision input
        bool            // both 32bit inputs
> InsertIdentityEltwiseTestParams;

class InsertIdentityLayerEltwiseTest: public InsertIdentityLayerTest,
                                      public ::testing::WithParamInterface<InsertIdentityEltwiseTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertIdentityEltwiseTestParams>& obj) {
        ELTWISE_TYPE type;
        bool low_precision, both_inputs_32bits;
        std::tie(type, low_precision, both_inputs_32bits) = obj.param;

        std::ostringstream result;
        result << "type=";
        switch (type) {
        case ELTWISE_TYPE::Sum:
            result << "sum";
            break;
        case ELTWISE_TYPE::Prod:
            result << "prod";
            break;
        default:
            break;
        }
        result << "_low_precision=" << low_precision;
        result << "_both_inputs_32bits=" << both_inputs_32bits;

        return result.str();
    }
    void SetUp() override {
        ELTWISE_TYPE type;
        bool both_inputs_32bits;
        std::tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            ngraph::ParameterVector params;
            auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto eltwise1 = std::make_shared<ngraph::op::Eltwise>(input1, const_input1, type);
            std::shared_ptr<ov::Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto eltwise2 = std::make_shared<ngraph::op::Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                second_input = const_input2;
            }

            auto eltwise3 = std::make_shared<ngraph::op::Eltwise>(eltwise1, second_input, type);

            auto result = std::make_shared<ngraph::opset9::Result>(eltwise3);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params});
        }

        {
            ngraph::ParameterVector params;
            auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto eltwise1 = std::make_shared<ngraph::op::Eltwise>(input1, const_input1, type);
            std::shared_ptr<ov::Node> first_input, second_input;
            first_input = eltwise1;

            if (both_inputs_32bits) {
                auto input2 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto eltwise2 = std::make_shared<ngraph::op::Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                second_input = const_input2;
            }

            if (type == ELTWISE_TYPE::Sum && !m_low_precision && both_inputs_32bits) {
                auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                first_input = identity;
            } else if (type == ELTWISE_TYPE::Prod || m_low_precision) {
                auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                first_input = identity;
                if (both_inputs_32bits) {
                    auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                    second_input = identity;
                }
            }

            auto eltwise3 = std::make_shared<ngraph::op::Eltwise>(first_input, second_input, type);

            auto result = std::make_shared<ngraph::opset9::Result>(eltwise3);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params});
        }
    }
};

TEST_P(InsertIdentityLayerEltwiseTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertIdentityLayerEltwiseTest,
                         ::testing::Combine(
                                ::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                ::testing::ValuesIn({true, false}),
                                ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseTest::getTestCaseName);

/******************************************* Eltwise layer tests (Multiple outputs) *************************************************/

class InsertIdentityLayerEltwiseMultipleOutputTest: public InsertIdentityLayerEltwiseTest {
public:
    void SetUp() override {
        ELTWISE_TYPE type;
        bool both_inputs_32bits;
        std::tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            ngraph::ParameterVector params;
            auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto eltwise1 = std::make_shared<ngraph::op::Eltwise>(input1, const_input1, type);
            std::shared_ptr<ov::Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto eltwise2 = std::make_shared<ngraph::op::Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                second_input = const_input2;
            }
            auto relu = std::make_shared<ngraph::opset9::Relu>(eltwise1);
            auto eltwise3 = std::make_shared<ngraph::op::Eltwise>(eltwise1, second_input, type);

            auto result1 = std::make_shared<ngraph::opset9::Result>(relu);
            auto result2 = std::make_shared<ngraph::opset9::Result>(eltwise3);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                        ngraph::ParameterVector{params});
        }

        {
            ngraph::ParameterVector params;
            auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto eltwise1 = std::make_shared<ngraph::op::Eltwise>(input1, const_input1, type);
            std::shared_ptr<ov::Node> first_input, second_input;
            first_input = eltwise1;

            if (both_inputs_32bits) {
                auto input2 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto eltwise2 = std::make_shared<ngraph::op::Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                second_input = const_input2;
            }

            if (type == ELTWISE_TYPE::Sum && !m_low_precision && both_inputs_32bits) {
                auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                first_input = identity;
            } else if (type == ELTWISE_TYPE::Prod || m_low_precision) {
                auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                first_input = identity;
                if (both_inputs_32bits) {
                    auto identity = std::make_shared<ov::intel_gna::op::Identity>(eltwise1);
                    second_input = identity;
                }
            }
            auto relu = std::make_shared<ngraph::opset9::Relu>(first_input);
            auto eltwise3 = std::make_shared<ngraph::op::Eltwise>(first_input, second_input, type);

            auto result1 = std::make_shared<ngraph::opset9::Result>(relu);
            auto result2 = std::make_shared<ngraph::opset9::Result>(eltwise3);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                            ngraph::ParameterVector{params});
        }
    }
};

TEST_P(InsertIdentityLayerEltwiseMultipleOutputTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertIdentityLayerEltwiseMultipleOutputTest,
                         ::testing::Combine(
                                ::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                ::testing::ValuesIn({true, false}),
                                ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseMultipleOutputTest::getTestCaseName);


/*************************************************** Eltwise with FQ layer tests ****************************************************/

class InsertIdentityLayerEltwiseFQTest: public InsertIdentityLayerEltwiseTest {
public:
    void SetUp() override {
        ELTWISE_TYPE type;
        bool both_inputs_32bits;
        std::tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();

        auto add_fake_quantize = [&](const std::shared_ptr<ngraph::Node>& node) {
            auto levels = (m_low_precision) ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max();
            auto input_low = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {5});
            auto output_low = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset9::FakeQuantize>(node, input_low, input_high, output_low, output_high, levels);
        };

        {
            ngraph::ParameterVector params;
            auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            params.push_back(input1);
            auto input1_fq = add_fake_quantize(input1);
            auto const_input1 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto const_input1_fq = add_fake_quantize(const_input1);
            auto eltwise1 = std::make_shared<ngraph::op::Eltwise>(input1_fq, const_input1_fq, type);
            auto eltwise1_fq = add_fake_quantize(eltwise1);
            std::shared_ptr<ov::Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
                params.push_back(input2);
                auto input2_fq = add_fake_quantize(input2);
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto const_input2_fq = add_fake_quantize(const_input2);
                auto eltwise2 = std::make_shared<ngraph::op::Eltwise>(input2_fq, const_input2_fq, type);
                auto eltwise2_fq = add_fake_quantize(eltwise2);
                second_input = eltwise2_fq;
            } else {
                auto const_input2 = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
                auto const_input2_fq = add_fake_quantize(const_input2);
                second_input = const_input2_fq;
            }

            auto eltwise3 = std::make_shared<ngraph::op::Eltwise>(eltwise1_fq, second_input, type);
            auto eltwise3_fq = add_fake_quantize(eltwise3);

            auto result = std::make_shared<ngraph::opset9::Result>(eltwise3_fq);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params});
        }

        {
            m_ref_func = m_func->clone();
        }
    }
};

TEST_P(InsertIdentityLayerEltwiseFQTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertIdentityLayerEltwiseFQTest,
                         ::testing::Combine(
                                ::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                ::testing::ValuesIn({true, false}),
                                ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseFQTest::getTestCaseName);

/***************************************************** Convolution layer tests *****************************************************/

typedef std::tuple<
        bool,           // with pooling
        bool,           // with activation
        bool            // swap matmul input
> InsertIdentityConvTestParams;

class InsertIdentityLayerConvMatMulTest: public InsertIdentityLayerTest,
                                         public ::testing::WithParamInterface<InsertIdentityConvTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertIdentityConvTestParams>& obj) {
        bool with_pool, with_act, swap_matmul;
        std::tie(with_pool, with_act, swap_matmul) = obj.param;

        std::ostringstream result;
        result << "with_pool=" << with_pool;
        result << "_with_act=" << with_act;
        result << "_swap_matmul=" << swap_matmul;

        return result.str();
    }
    void SetUp() override {
        bool with_pool, with_act, swap_matmul;
        std::tie(with_pool, with_act, swap_matmul) = this->GetParam();

        InsertIdentityLayerTest::SetUp();

        m_input_shape = {1, 3, 1, 64};
        auto reshape_shape = ngraph::Shape{3, 64};

        {
            std::shared_ptr<ov::Node> last_node;
            auto input = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto weights = ngraph::opset9::Constant::create(ngraph::element::f32,
                                                            ngraph::Shape{3, 3, 1, 2}, {1});
            auto conv = std::make_shared<ngraph::opset9::Convolution>(input, weights,
                                                                      ngraph::Strides{1, 1},
                                                                      ngraph::CoordinateDiff{0, 0},
                                                                      ngraph::CoordinateDiff{0, 1},
                                                                      ngraph::Strides{1, 1});
            last_node = conv;
            if (with_pool) {
                auto max_pool = std::make_shared<ngraph::opset7::MaxPool>(last_node,
                                                                          ngraph::Strides{1, 1},
                                                                          ngraph::Shape{0, 0},
                                                                          ngraph::Shape{0, 1},
                                                                          ngraph::Shape{1, 2});
                last_node = max_pool;
            }
            if (with_act) {
                auto relu = std::make_shared<ngraph::opset9::Relu>(last_node);
                last_node = relu;
            }
            auto reshape_const = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{reshape_shape.size()}, reshape_shape);
            auto reshape = std::make_shared<ngraph::opset9::Reshape>(last_node, reshape_const, false);
            auto matmul_const = ngraph::opset9::Constant::create(ngraph::element::f32, {64, 3}, {1.2});
            auto matmul = swap_matmul ? std::make_shared<ngraph::opset9::MatMul>(matmul_const, reshape) :
                                        std::make_shared<ngraph::opset9::MatMul>(reshape, matmul_const);

            auto result = std::make_shared<ngraph::opset9::Result>(matmul);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{input});
        }

        {
            std::shared_ptr<ov::Node> last_node;
            auto input = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto weights = ngraph::opset9::Constant::create(ngraph::element::f32,
                                                            ngraph::Shape{3, 3, 1, 2}, {1});
            auto conv = std::make_shared<ngraph::opset9::Convolution>(input, weights,
                                                                       ngraph::Strides{1, 1},
                                                                       ngraph::CoordinateDiff{0, 0},
                                                                       ngraph::CoordinateDiff{0, 1},
                                                                       ngraph::Strides{1, 1});
            last_node = conv;
            if (with_pool) {
                auto max_pool = std::make_shared<ngraph::opset7::MaxPool>(last_node,
                                                                          ngraph::Strides{1, 1},
                                                                          ngraph::Shape{0, 0},
                                                                          ngraph::Shape{0, 1},
                                                                          ngraph::Shape{1, 2});
                last_node = max_pool;
            }
            if (with_act) {
                auto relu = std::make_shared<ngraph::opset9::Relu>(last_node);
                last_node = relu;
            } else {
                auto identity = std::make_shared<ov::intel_gna::op::Identity>(last_node);
                last_node = identity;
            }
            auto reshape_const = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{reshape_shape.size()}, reshape_shape);
            auto reshape = std::make_shared<ngraph::opset9::Reshape>(last_node, reshape_const, false);
            auto matmul_const = ngraph::opset9::Constant::create(ngraph::element::f32, {64, 3}, {1.2});
            auto matmul = swap_matmul ? std::make_shared<ngraph::opset9::MatMul>(matmul_const, reshape) :
                                        std::make_shared<ngraph::opset9::MatMul>(reshape, matmul_const);

            auto result = std::make_shared<ngraph::opset9::Result>(matmul);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{input});
        }
    }
};

TEST_P(InsertIdentityLayerConvMatMulTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertIdentityLayerConvMatMulTest,
                         ::testing::Combine(
                                ::testing::ValuesIn({true, false}),
                                ::testing::ValuesIn({true, false}),
                                ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerConvMatMulTest::getTestCaseName);

/***************************************************** Result layer tests *****************************************************/

class InsertIdentityLayerResultTest: public InsertIdentityLayerTest {
public:
    void SetUp() override {
        InsertIdentityLayerTest::SetUp();
        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            auto relu = std::make_shared<ngraph::opset9::Relu>(add);
            auto result1 = std::make_shared<ngraph::opset9::Result>(add);
            auto result2 = std::make_shared<ngraph::opset9::Result>(relu);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                        ngraph::ParameterVector{params});
        }

        {
            auto params = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, m_input_shape);
            auto const_add = ngraph::opset9::Constant::create(ngraph::element::f32, m_input_shape, {1});
            auto add = std::make_shared<ngraph::opset9::Add>(params, const_add);
            auto identity = std::make_shared<ov::intel_gna::op::Identity>(add);
            auto relu = std::make_shared<ngraph::opset9::Relu>(add);
            auto result1 = std::make_shared<ngraph::opset9::Result>(identity);
            auto result2 = std::make_shared<ngraph::opset9::Result>(relu);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                            ngraph::ParameterVector{params});
        }
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::BreakFusingOfOutputLayers>();
        m.run_passes(m_func);
        ASSERT_NO_THROW(check_rt_info(m_func));

        auto result = compare_functions(m_func, m_ref_func);
        ASSERT_TRUE(result.first);
    }
};

TEST_F(InsertIdentityLayerResultTest, CompareWithRefs) {
    Run();
}
} // namespace testing
