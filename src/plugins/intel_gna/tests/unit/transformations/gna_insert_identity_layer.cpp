// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ov_test_utils.hpp>
#include <legacy/ngraph_ops/eltwise.hpp>
#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_models/builders.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset7.hpp"
#include "ops/identity.hpp"
#include "transformations/insert_identity_layer.hpp"
#include "transformations/rt_info/gna_precision_change_flag.hpp"

namespace testing {

using namespace ngraph::builder;
using namespace ngraph::op;
using namespace ov;
using namespace ov::opset12;
using namespace ov::pass;
using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::op;
using namespace ov::element;
using namespace std;

class InsertIdentityLayerTest : public ov::test::TestsCommon {
public:
    virtual void Validate();
    virtual void Run();

public:
    shared_ptr<Model> m_func, m_ref_func;
    Shape m_input_shape{10};
    bool m_low_precision = false;
};

void InsertIdentityLayerTest::Validate() {
    Manager m;
    m.register_pass<InitNodeInfo>();
    m.register_pass<MarkIdentityCandidates>(m_low_precision);
    m.register_pass<InsertIdentity>();
    m.register_pass<BreakFusingOfOutputLayers>();
    m.run_passes(m_func);
    ASSERT_NO_THROW(check_rt_info(m_func));

    auto result = compare_functions(m_func, m_ref_func);
    ASSERT_TRUE(result.first);

    // Cleanup rt info and check
    m.register_pass<IdentityCandidatesCleanup>();
    m.run_passes(m_func);
    for (auto& node : m_func->get_ordered_ops()) {
        for (auto& input : node->inputs()) {
            const RTMap& rt_info = input.get_rt_info();
            ASSERT_EQ(rt_info.count(rt_info::GNAPrecisionChangeFlag::get_type_info_static()), 0);
        }
    }
}

void InsertIdentityLayerTest::Run() {
    SetUp();
    Validate();
}

/******************************************************* Concat layer tests
 * *******************************************************/

typedef tuple<size_t,  // Concat axis
              size_t   // input number
              >
    InsertIdentityConcatTestParams;

class InsertIdentityLayerConcatTest : public InsertIdentityLayerTest,
                                      public ::testing::WithParamInterface<InsertIdentityConcatTestParams> {
public:
    static string getTestCaseName(const testing::TestParamInfo<InsertIdentityConcatTestParams>& obj) {
        size_t axis, inputs_num;
        tie(axis, inputs_num) = obj.param;

        ostringstream result;
        result << "inputsNum=" << inputs_num << "_";
        result << "axis=" << axis;

        return result.str();
    }
    void SetUp() override {
        size_t axis, inputs_num;
        tie(axis, inputs_num) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            OutputVector concat_inputs = {add};
            for (size_t i = 1; i < inputs_num; ++i) {
                auto const_mul = Constant::create(f32, m_input_shape, {i});
                auto mul = make_shared<Multiply>(add, const_mul);
                concat_inputs.push_back(mul);
            }
            auto concat = make_shared<Concat>(concat_inputs, axis);
            auto result = make_shared<Result>(concat);
            m_func = make_shared<Model>(ResultVector{result}, ParameterVector{params});
        }

        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            auto identity = make_shared<Identity>(add);
            OutputVector concat_inputs = {identity};
            for (size_t i = 1; i < inputs_num; ++i) {
                auto const_mul = Constant::create(f32, m_input_shape, {i});
                auto mul = make_shared<Multiply>(identity, const_mul);
                auto identity_mul = make_shared<Identity>(mul);
                concat_inputs.push_back(identity_mul);
            }
            auto concat = make_shared<Concat>(concat_inputs, axis);
            auto result = make_shared<Result>(concat);
            m_ref_func = make_shared<Model>(ResultVector{result}, ParameterVector{params});
        }
    }
};

const size_t axis = 0;
const vector<size_t> inputCounts = {1, 8};

TEST_P(InsertIdentityLayerConcatTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertIdentityLayerConcatTest,
                         ::testing::Combine(::testing::Values(axis), ::testing::ValuesIn(inputCounts)),
                         InsertIdentityLayerConcatTest::getTestCaseName);

/******************************************************* Split layer tests
 * *******************************************************/

class InsertIdentityLayerSplitTest : public InsertIdentityLayerTest {
public:
    void SetUp() override {
        InsertIdentityLayerTest::SetUp();
        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            auto axis_const = Constant::create(i64, Shape{}, {0});
            auto split = make_shared<Split>(add, axis_const, 2);
            auto result1 = make_shared<Result>(split->output(0));
            auto const_reshape = Constant::create(i64, {2}, {1, 5});
            auto reshape = make_shared<Reshape>(split->output(1), const_reshape, false);
            auto const_mul = Constant::create(f32, {1, 5}, {1});
            auto mul = make_shared<Multiply>(reshape, const_mul);
            auto result2 = make_shared<Result>(mul);
            m_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }

        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            auto identity = make_shared<Identity>(add);
            auto axis_const = Constant::create(i64, Shape{}, {0});
            auto split = make_shared<Split>(identity, axis_const, 2);
            auto result1 = make_shared<Result>(split->output(0));
            auto const_reshape = Constant::create(i64, {2}, {1, 5});
            auto reshape = make_shared<Reshape>(split->output(1), const_reshape, false);
            auto const_mul = Constant::create(f32, {1, 5}, {1});
            auto mul = make_shared<Multiply>(reshape, const_mul);
            auto result2 = make_shared<Result>(mul);
            m_ref_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }
    }
};

TEST_F(InsertIdentityLayerSplitTest, CompareWithRefs) {
    Run();
}

/******************************************************* Eltwise layer tests
 * *******************************************************/

typedef tuple<ELTWISE_TYPE,  // eltwise type
              bool,          // use low precision input
              bool           // both 32bit inputs
              >
    InsertIdentityEltwiseTestParams;

class InsertIdentityLayerEltwiseTest : public InsertIdentityLayerTest,
                                       public ::testing::WithParamInterface<InsertIdentityEltwiseTestParams> {
public:
    static string getTestCaseName(const testing::TestParamInfo<InsertIdentityEltwiseTestParams>& obj) {
        ELTWISE_TYPE type;
        bool low_precision, both_inputs_32bits;
        tie(type, low_precision, both_inputs_32bits) = obj.param;

        ostringstream result;
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
        tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            ParameterVector params;
            auto input1 = make_shared<Parameter>(f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = Constant::create(f32, m_input_shape, {1});
            auto eltwise1 = make_shared<Eltwise>(input1, const_input1, type);
            shared_ptr<Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = make_shared<Parameter>(f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto eltwise2 = make_shared<Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                second_input = const_input2;
            }

            auto eltwise3 = make_shared<Eltwise>(eltwise1, second_input, type);

            auto result = make_shared<Result>(eltwise3);
            m_func = make_shared<Model>(ResultVector{result}, ParameterVector{params});
        }

        {
            ParameterVector params;
            auto input1 = make_shared<Parameter>(f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = Constant::create(f32, m_input_shape, {1});
            auto eltwise1 = make_shared<Eltwise>(input1, const_input1, type);
            shared_ptr<Node> first_input, second_input;
            first_input = eltwise1;

            if (both_inputs_32bits) {
                auto input2 = make_shared<Parameter>(f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto eltwise2 = make_shared<Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                second_input = const_input2;
            }

            if (type == ELTWISE_TYPE::Sum && !m_low_precision && both_inputs_32bits) {
                auto identity = make_shared<Identity>(eltwise1);
                first_input = identity;
            } else if (type == ELTWISE_TYPE::Prod || m_low_precision) {
                auto identity = make_shared<Identity>(eltwise1);
                first_input = identity;
                if (both_inputs_32bits) {
                    auto identity = make_shared<Identity>(eltwise1);
                    second_input = identity;
                }
            }

            auto eltwise3 = make_shared<Eltwise>(first_input, second_input, type);

            auto result = make_shared<Result>(eltwise3);
            m_ref_func = make_shared<Model>(ResultVector{result}, ParameterVector{params});
        }
    }
};

TEST_P(InsertIdentityLayerEltwiseTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertIdentityLayerEltwiseTest,
                         ::testing::Combine(::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseTest::getTestCaseName);

/******************************************* Eltwise layer tests (Multiple outputs)
 * *************************************************/

class InsertIdentityLayerEltwiseMultipleOutputTest : public InsertIdentityLayerEltwiseTest {
public:
    void SetUp() override {
        ELTWISE_TYPE type;
        bool both_inputs_32bits;
        tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();
        {
            ParameterVector params;
            auto input1 = make_shared<Parameter>(f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = Constant::create(f32, m_input_shape, {1});
            auto eltwise1 = make_shared<Eltwise>(input1, const_input1, type);
            shared_ptr<Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = make_shared<Parameter>(f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto eltwise2 = make_shared<Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                second_input = const_input2;
            }
            auto relu = make_shared<Relu>(eltwise1);
            auto eltwise3 = make_shared<Eltwise>(eltwise1, second_input, type);

            auto result1 = make_shared<Result>(relu);
            auto result2 = make_shared<Result>(eltwise3);
            m_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }

        {
            ParameterVector params;
            auto input1 = make_shared<Parameter>(f32, m_input_shape);
            params.push_back(input1);
            auto const_input1 = Constant::create(f32, m_input_shape, {1});
            auto eltwise1 = make_shared<Eltwise>(input1, const_input1, type);
            shared_ptr<Node> first_input, second_input;
            first_input = eltwise1;

            if (both_inputs_32bits) {
                auto input2 = make_shared<Parameter>(f32, m_input_shape);
                params.push_back(input2);
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto eltwise2 = make_shared<Eltwise>(input2, const_input2, type);
                second_input = eltwise2;
            } else {
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                second_input = const_input2;
            }

            if (type == ELTWISE_TYPE::Sum && !m_low_precision && both_inputs_32bits) {
                auto identity = make_shared<Identity>(eltwise1);
                first_input = identity;
            } else if (type == ELTWISE_TYPE::Prod || m_low_precision) {
                auto identity = make_shared<Identity>(eltwise1);
                first_input = identity;
                if (both_inputs_32bits) {
                    auto identity = make_shared<Identity>(eltwise1);
                    second_input = identity;
                }
            }
            auto relu = make_shared<Relu>(first_input);
            auto eltwise3 = make_shared<Eltwise>(first_input, second_input, type);

            auto result1 = make_shared<Result>(relu);
            auto result2 = make_shared<Result>(eltwise3);
            m_ref_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }
    }
};

TEST_P(InsertIdentityLayerEltwiseMultipleOutputTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertIdentityLayerEltwiseMultipleOutputTest,
                         ::testing::Combine(::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseMultipleOutputTest::getTestCaseName);

/*************************************************** Eltwise with FQ layer tests
 * ****************************************************/

class InsertIdentityLayerEltwiseFQTest : public InsertIdentityLayerEltwiseTest {
public:
    void SetUp() override {
        ELTWISE_TYPE type;
        bool both_inputs_32bits;
        tie(type, m_low_precision, both_inputs_32bits) = this->GetParam();

        InsertIdentityLayerTest::SetUp();

        auto add_fake_quantize = [&](const shared_ptr<Node>& node) {
            auto levels = (m_low_precision) ? numeric_limits<int8_t>::max() : numeric_limits<int16_t>::max();
            auto input_low = Constant::create(i64, Shape{1}, {1});
            auto input_high = Constant::create(i64, Shape{1}, {5});
            auto output_low = Constant::create(i64, Shape{1}, {0});
            auto output_high = Constant::create(i64, Shape{1}, {10});
            return make_shared<FakeQuantize>(node, input_low, input_high, output_low, output_high, levels);
        };

        {
            ParameterVector params;
            auto input1 = make_shared<Parameter>(f32, m_input_shape);
            params.push_back(input1);
            auto input1_fq = add_fake_quantize(input1);
            auto const_input1 = Constant::create(f32, m_input_shape, {1});
            auto const_input1_fq = add_fake_quantize(const_input1);
            auto eltwise1 = make_shared<Eltwise>(input1_fq, const_input1_fq, type);
            auto eltwise1_fq = add_fake_quantize(eltwise1);
            shared_ptr<Node> second_input;

            if (both_inputs_32bits) {
                auto input2 = make_shared<Parameter>(f32, m_input_shape);
                params.push_back(input2);
                auto input2_fq = add_fake_quantize(input2);
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto const_input2_fq = add_fake_quantize(const_input2);
                auto eltwise2 = make_shared<Eltwise>(input2_fq, const_input2_fq, type);
                auto eltwise2_fq = add_fake_quantize(eltwise2);
                second_input = eltwise2_fq;
            } else {
                auto const_input2 = Constant::create(f32, m_input_shape, {1});
                auto const_input2_fq = add_fake_quantize(const_input2);
                second_input = const_input2_fq;
            }

            auto eltwise3 = make_shared<Eltwise>(eltwise1_fq, second_input, type);
            auto eltwise3_fq = add_fake_quantize(eltwise3);

            auto result = make_shared<Result>(eltwise3_fq);
            m_func = make_shared<Model>(ResultVector{result}, ParameterVector{params});
        }

        { m_ref_func = m_func->clone(); }
    }
};

TEST_P(InsertIdentityLayerEltwiseFQTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertIdentityLayerEltwiseFQTest,
                         ::testing::Combine(::testing::ValuesIn({ELTWISE_TYPE::Sum, ELTWISE_TYPE::Prod}),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerEltwiseFQTest::getTestCaseName);

/***************************************************** Convolution layer tests
 * *****************************************************/

typedef tuple<bool,  // with pooling
              bool,  // with activation
              bool   // swap matmul input
              >
    InsertIdentityConvTestParams;

class InsertIdentityLayerConvMatMulTest : public InsertIdentityLayerTest,
                                          public ::testing::WithParamInterface<InsertIdentityConvTestParams> {
public:
    static string getTestCaseName(const testing::TestParamInfo<InsertIdentityConvTestParams>& obj) {
        bool with_pool, with_act, swap_matmul;
        tie(with_pool, with_act, swap_matmul) = obj.param;

        ostringstream result;
        result << "with_pool=" << with_pool;
        result << "_with_act=" << with_act;
        result << "_swap_matmul=" << swap_matmul;

        return result.str();
    }
    void SetUp() override {
        bool with_pool, with_act, swap_matmul;
        tie(with_pool, with_act, swap_matmul) = this->GetParam();

        InsertIdentityLayerTest::SetUp();

        m_input_shape = {1, 3, 1, 64};
        auto reshape_shape = Shape{3, 64};

        {
            shared_ptr<Node> last_node;
            auto input = make_shared<Parameter>(f32, m_input_shape);
            auto weights = Constant::create(f32, Shape{3, 3, 1, 2}, {1});
            auto conv = make_shared<Convolution>(input,
                                                 weights,
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 1},
                                                 Strides{1, 1});
            last_node = conv;
            if (with_pool) {
                auto max_pool =
                    make_shared<opset7::MaxPool>(last_node, Strides{1, 1}, Shape{0, 0}, Shape{0, 1}, Shape{1, 2});
                last_node = max_pool;
            }
            if (with_act) {
                auto relu = make_shared<Relu>(last_node);
                last_node = relu;
            }
            auto reshape_const = Constant::create(i64, Shape{reshape_shape.size()}, reshape_shape);
            auto reshape = make_shared<Reshape>(last_node, reshape_const, false);
            auto matmul_const = Constant::create(f32, {64, 3}, {1.2});
            auto matmul =
                swap_matmul ? make_shared<MatMul>(matmul_const, reshape) : make_shared<MatMul>(reshape, matmul_const);

            auto result = make_shared<Result>(matmul);
            m_func = make_shared<Model>(ResultVector{result}, ParameterVector{input});
        }

        {
            shared_ptr<Node> last_node;
            auto input = make_shared<Parameter>(f32, m_input_shape);
            auto weights = Constant::create(f32, Shape{3, 3, 1, 2}, {1});
            auto conv = make_shared<Convolution>(input,
                                                 weights,
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 1},
                                                 Strides{1, 1});
            last_node = conv;
            if (with_pool) {
                auto max_pool =
                    make_shared<opset7::MaxPool>(last_node, Strides{1, 1}, Shape{0, 0}, Shape{0, 1}, Shape{1, 2});
                last_node = max_pool;
            }
            if (with_act) {
                auto relu = make_shared<Relu>(last_node);
                last_node = relu;
            } else {
                auto identity = make_shared<Identity>(last_node);
                last_node = identity;
            }
            auto reshape_const = Constant::create(i64, Shape{reshape_shape.size()}, reshape_shape);
            auto reshape = make_shared<Reshape>(last_node, reshape_const, false);
            auto matmul_const = Constant::create(f32, {64, 3}, {1.2});
            auto matmul =
                swap_matmul ? make_shared<MatMul>(matmul_const, reshape) : make_shared<MatMul>(reshape, matmul_const);

            auto result = make_shared<Result>(matmul);
            m_ref_func = make_shared<Model>(ResultVector{result}, ParameterVector{input});
        }
    }
};

TEST_P(InsertIdentityLayerConvMatMulTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         InsertIdentityLayerConvMatMulTest,
                         ::testing::Combine(::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::ValuesIn({true, false})),
                         InsertIdentityLayerConvMatMulTest::getTestCaseName);

/***************************************************** Result layer tests
 * *****************************************************/

class InsertIdentityLayerResultTest : public InsertIdentityLayerTest {
public:
    void SetUp() override {
        InsertIdentityLayerTest::SetUp();
        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            auto relu = make_shared<Relu>(add);
            auto result1 = make_shared<Result>(add);
            auto result2 = make_shared<Result>(relu);
            m_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }

        {
            auto params = make_shared<Parameter>(f32, m_input_shape);
            auto const_add = Constant::create(f32, m_input_shape, {1});
            auto add = make_shared<Add>(params, const_add);
            auto identity = make_shared<Identity>(add);
            auto relu = make_shared<Relu>(add);
            auto result1 = make_shared<Result>(identity);
            auto result2 = make_shared<Result>(relu);
            m_ref_func = make_shared<Model>(ResultVector{result1, result2}, ParameterVector{params});
        }
    }
    void Validate() override {
        Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<BreakFusingOfOutputLayers>();
        m.run_passes(m_func);
        ASSERT_NO_THROW(check_rt_info(m_func));

        auto result = compare_functions(m_func, m_ref_func);
        ASSERT_TRUE(result.first);
    }
};

TEST_F(InsertIdentityLayerResultTest, CompareWithRefs) {
    Run();
}

class InsertIdentityForNonQuantizableConcatInputTest : public InsertIdentityLayerTest {
    string getName() {
        return "InsertIdentityForPrecAgnosticConcatInput";
    }

    shared_ptr<FakeQuantize> create_fq(const Type& type,
                                       const shared_ptr<ov::Node>& node,
                                       float fq_min,
                                       float fq_max,
                                       std::size_t levels) {
        //
        auto fq_inp_min = makeConstant<float>(type, {1}, {fq_min});
        auto fq_inp_max = makeConstant<float>(type, {1}, {fq_max});
        auto fq_out_min = makeConstant<float>(type, {1}, {fq_min});
        auto fq_out_max = makeConstant<float>(type, {1}, {fq_max});
        return make_shared<FakeQuantize>(node, fq_inp_min, fq_inp_max, fq_out_min, fq_out_max, levels);
    }

public:
    void SetUp() override {
        InsertIdentityLayerTest::SetUp();
        {
            ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_input_shape),
                                       std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_input_shape)};
            auto fq = create_fq(f32, inputs[0], -1, 1, 256);
            auto relu = make_shared<Relu>(fq);
            auto reshape_const = make_shared<Constant>(i64, Shape{1}, m_input_shape);
            auto reshape = make_shared<Reshape>(inputs[1], reshape_const, false);
            auto concat = std::make_shared<ov::op::v0::Concat>({relu, reshape}, 0);
            auto result = make_shared<Result>(concat);
            m_func = make_shared<Model>(result, inputs, getName());
        }

        {
            ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_input_shape),
                                       std::make_shared<ov::op::v0::Parameter>(ov::element::f32, m_input_shape)};
            auto fq = create_fq(f32, inputs[0], -1, 1, 256);
            auto relu = make_shared<Relu>(fq);
            auto reshape_const = make_shared<Constant>(i64, Shape{1}, m_input_shape);
            auto reshape = make_shared<Reshape>(inputs[1], reshape_const, false);
            // We expect the following Identity layer to be inserted
            auto identity = make_shared<Identity>(reshape);
            auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{relu, identity}, 0);
            auto result = make_shared<Result>(concat);
            m_ref_func = make_shared<Model>(result, inputs, getName());
        }
    }

    void Validate() override {
        Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<InsertIdentityForPrecAgnosticConcatInput>();
        m.run_passes(m_func);
        ASSERT_NO_THROW(check_rt_info(m_func));

        auto result = compare_functions(m_func, m_ref_func);
        ASSERT_TRUE(result.first);
    }
};

TEST_F(InsertIdentityForNonQuantizableConcatInputTest, CompareWithRefs) {
    Run();
}
}  // namespace testing
