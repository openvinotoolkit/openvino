// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/node.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/reduce_logical_and.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/round.hpp"
#include "ngraph/op/scatter_elements_update.hpp"
#include "ngraph/op/scatter_update.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/unsqueeze.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

#define ASSERT_FLOAT_VECTORS_EQ(expected, result)                                                  \
    ASSERT_EQ(expected.size(), result.size()) << "Array sizes differ.";                            \
    for (size_t i = 0; i < expected.size(); ++i)                                                   \
    {                                                                                              \
        ASSERT_FLOAT_EQ(expected[i], result[i]) << "at index: " << i;                              \
    }

TEST(eval, bad_get_data_ptr)
{
    HostTensor c(element::f32, Shape{});
    *c.get_data_ptr<float>() = 1.0;
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::f32>(), 1.0);
    try
    {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try
    {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}

TEST(eval, max_eval_parameter)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{});

    auto result = maximum_value(p);
    EXPECT_FALSE(result.first);
    EXPECT_EQ(result.second, numeric_limits<uint64_t>::max());
}

TEST(eval, max_eval_constant)
{
    auto c = op::Constant::create<int64_t>(element::i64, Shape{}, {27});
    auto result = maximum_value(c);
    ASSERT_TRUE(result.first);
    EXPECT_EQ(result.second, 27);
}

TEST(eval, max_eval_minimum_constant)
{
    auto c = op::Constant::create<int64_t>(element::i64, Shape{}, {27});
    auto p = make_shared<op::Parameter>(element::i64, Shape{});
    auto m = make_shared<op::v1::Minimum>(c, p);
    auto result = maximum_value(m);
    ASSERT_TRUE(result.first);
    EXPECT_EQ(result.second, 27);
}

TEST(eval, max_eval_reduce_min)
{
    auto concat = make_shared<op::v0::Convert>(
        make_shared<op::v0::Concat>(
            OutputVector{make_shared<op::v0::Parameter>(element::i64, Shape{4}),
                         make_shared<op::v0::Constant>(element::i64, Shape{4}, 37)},
            0),
        element::i32);
    auto reduce = make_shared<op::v0::Convert>(
        make_shared<op::v1::ReduceMin>(concat,
                                       make_shared<op::v0::Constant>(element::i32, Shape{1}, 0)),
        element::i64);
    auto squeezes = make_shared<op::v0::Squeeze>(
        make_shared<op::v0::Unsqueeze>(reduce,
                                       make_shared<op::v0::Constant>(element::i32, Shape{1}, 0)),
        make_shared<op::v0::Constant>(element::i64, Shape{1}, 0));
    EXPECT_EQ(maximum_value(squeezes).second, 37);
}

TEST(eval, evaluate_shape_of)
{
    auto p = make_shared<op::Parameter>(element::f32, PartialShape{-1, -1});
    auto so = make_shared<op::v0::ShapeOf>(p);
    auto fun = make_shared<Function>(OutputVector{so}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2}));
    auto result_shape = read_vector<int64_t>(result);
    vector<int64_t> arg_shape{2, 3};
    ASSERT_EQ(result_shape, arg_shape);
}

TEST(eval, evaluate_dynamic_range_sum)
{
    auto p_start = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_stop = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_step = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto range = make_shared<op::v0::Range>(p_start, p_stop, p_step);
    auto add = make_shared<op::v1::Add>(range, p1);
    auto fun =
        make_shared<Function>(OutputVector{add}, ParameterVector{p_start, p_stop, p_step, p1});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({}, {1.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {10.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {3.0f}),
                               make_host_tensor<element::Type_t::f32>({}, {7.0f})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(cval, seq);
}

#ifdef NGRAPH_INTERPRETER_ENABLE
TEST(eval, interpret_dynamic_range_sum)
{
    auto p_start = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_stop = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p_step = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto p1 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto range = make_shared<op::v0::Range>(p_start, p_stop, p_step);
    auto add = make_shared<op::v1::Add>(range, p1);
    auto fun =
        make_shared<Function>(OutputVector{add}, ParameterVector{p_start, p_stop, p_step, p1});
    auto backend = runtime::Backend::create("INTERPRETER");
    auto p_start_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_start_val, vector<float>{1.0f});
    auto p_stop_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_stop_val, vector<float>{10.0f});
    auto p_step_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p_step_val, vector<float>{3.0f});
    auto p1_val = backend->create_tensor(element::f32, Shape{});
    copy_data(p1_val, vector<float>{7.0f});
    auto result = backend->create_tensor();
    auto cfun = backend->compile(fun);
    cfun->call({result}, {p_start_val, p_stop_val, p_step_val, p1_val});
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{3}));
    auto result_val = read_vector<float>(result);
    vector<float> seq{8.0f, 11.0f, 14.0f};
    ASSERT_EQ(result_val, seq);
}
#endif

TEST(eval, evaluate_broadcast_v3_bidirectional)
{
    Shape shape_a{4, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int32_t>(element::i32, Shape{3}, {2, 1, 4});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{4, 1}, {1.0f, 2.0f, 3.0f, 4.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 4, 4}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_bidirectional_target_rank_smaller_than_input)
{
    Shape shape_a{1, 1, 1, 1, 1, 1, 1, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{4}, {1, 3, 1, 1});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(shape_a, {1.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{1, 1, 1, 1, 1, 3, 1, 1}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1.0f, 1.0f, 1.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_bidirectional_target_rank_smaller_than_input_2)
{
    Shape shape_a{1, 3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int32_t>(element::i32, Shape{2}, {3, 1});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{1, 3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{1, 3, 1}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1.0f, 2.0f, 3.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_bidirectional_dyn)
{
    Shape shape_a{4, 1};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i32, Shape{3});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A, target_shape});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(Shape{4, 1}, {1, 2, 3, 4}),
                               make_host_tensor<element::Type_t::i32>(Shape{3}, {2, 1, 4})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 4, 4}));
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                          1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_numpy)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});
    auto bcast_v3 = make_shared<op::v3::Broadcast>(A, target_shape);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_numpy_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i32, Shape{3});
    auto bcast_v3 = make_shared<op::v3::Broadcast>(A, target_shape);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A, target_shape});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i32>(Shape{3}, {2, 3, 6})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_numpy_vs_bidi)
{
    Shape in_shape{1, 4, 1};

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {1, 4, 4});
    auto bcast_v3_num = make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::NUMPY);
    auto fun_num = make_shared<Function>(OutputVector{bcast_v3_num}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun_num->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(in_shape, {1.0f, 2.0f, 3.0f, 4.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{1, 4, 4}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    ASSERT_EQ(expec, result_val);

    auto target_shape2 = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 4});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape2, op::BroadcastType::BIDIRECTIONAL);
    auto fun_bidi = make_shared<Function>(OutputVector{bcast_v3_num}, ParameterVector{A});

    auto result2 = make_shared<HostTensor>();
    ASSERT_TRUE(fun_bidi->evaluate(
        {result2}, {make_host_tensor<element::Type_t::f32>(in_shape, {1.0f, 2.0f, 3.0f, 4.0f})}));
    EXPECT_EQ(result2->get_element_type(), element::f32);
    EXPECT_EQ(result2->get_partial_shape(), (PartialShape{1, 4, 4}));
    auto result_val2 = read_vector<float>(result2);
    vector<float> expec2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    ASSERT_EQ(expec2, result_val2);
}

TEST(eval, evaluate_broadcast_v3_bidi_3d)
{
    Shape in_shape{1, 4, 1};

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {1, 1, 3});
    auto bcast_v3_num =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun_num = make_shared<Function>(OutputVector{bcast_v3_num}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun_num->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(in_shape, {1.0f, 2.0f, 3.0f, 4.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{1, 4, 3}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};
    ASSERT_EQ(expec, result_val);
}

TEST(eval, evaluate_broadcast_v3_bidi_4d)
{
    Shape in_shape{4, 1, 1};
    Shape expec_shape{1, 4, 2, 2};

    auto A = make_shared<op::Parameter>(element::f32, in_shape);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{4}, {1, 1, 2, 2});
    auto bcast_v3 =
        make_shared<op::v3::Broadcast>(A, target_shape, op::BroadcastType::BIDIRECTIONAL);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(in_shape, {1.0f, 2.0f, 3.0f, 4.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{1, 4, 2, 2}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_pdpd)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});
    auto bcast_v3 = make_shared<op::v3::Broadcast>(
        A, target_shape, op::BroadcastModeSpec(op::BroadcastType::PDPD, 1));
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_pdpd_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i32, Shape{3});
    auto bcast_v3 = make_shared<op::v3::Broadcast>(
        A, target_shape, op::BroadcastModeSpec(op::BroadcastType::PDPD, 1));
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A, target_shape});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i32>(Shape{3}, {2, 3, 6})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_numpy)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});
    auto bcast_v3 = make_shared<op::v1::Broadcast>(A, target_shape);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_numpy_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto bcast_v3 = make_shared<op::v1::Broadcast>(A, target_shape);
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A, target_shape});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i64>(Shape{3}, {2, 3, 6})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_pdpd)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 6});
    auto bcast_v3 = make_shared<op::v1::Broadcast>(
        A, target_shape, op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_pdpd_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto bcast_v3 = make_shared<op::v1::Broadcast>(
        A, target_shape, op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1));
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A, target_shape});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i64>(Shape{3}, {2, 3, 6})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 6}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    };
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_explicit)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 1});
    auto axes_mapping = op::Constant::create<int32_t>(element::i32, Shape{2}, {1, 2});
    auto bcast_v3 = make_shared<op::v1::Broadcast>(
        A, target_shape, axes_mapping, op::AutoBroadcastSpec(op::AutoBroadcastType::EXPLICIT));
    auto fun = make_shared<Function>(OutputVector{bcast_v3}, ParameterVector{A});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 1}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 2, 3, 1, 2, 3};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v1_explicit_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto axes_mapping = make_shared<op::Parameter>(element::i32, Shape{2});

    auto bcast_v1 = make_shared<op::v1::Broadcast>(
        A, target_shape, axes_mapping, op::AutoBroadcastSpec(op::AutoBroadcastType::EXPLICIT));
    auto fun = make_shared<Function>(OutputVector{bcast_v1},
                                     ParameterVector{A, target_shape, axes_mapping});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i64>(Shape{3}, {2, 3, 1}),
                       make_host_tensor<element::Type_t::i32>(Shape{2}, {1, 2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 1}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 2, 3, 1, 2, 3};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_broadcast_v3_explicit_dyn)
{
    Shape shape_a{3, 1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto target_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto axes_mapping = make_shared<op::Parameter>(element::i32, Shape{2});

    auto bcast_v3 = make_shared<op::v3::Broadcast>(
        A, target_shape, axes_mapping, op::BroadcastModeSpec(op::BroadcastType::EXPLICIT));
    auto fun = make_shared<Function>(OutputVector{bcast_v3},
                                     ParameterVector{A, target_shape, axes_mapping});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(Shape{3, 1}, {1.0f, 2.0f, 3.0f}),
                       make_host_tensor<element::Type_t::i64>(Shape{3}, {2, 3, 1}),
                       make_host_tensor<element::Type_t::i32>(Shape{2}, {1, 2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3, 1}));
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 2, 3, 1, 2, 3};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, test_op_multi_out)
{
    auto p = make_shared<op::Parameter>(element::f32, PartialShape{2, 3});
    auto p2 = make_shared<op::Parameter>(element::f64, PartialShape{2, 2});
    auto so = make_shared<TestOpMultiOut>(p, p2);
    auto fun =
        make_shared<Function>(OutputVector{so->output(0), so->output(1)}, ParameterVector{p, p2});
    auto result = make_shared<HostTensor>(element::Type_t::f32, Shape{2, 3});
    auto result2 = make_shared<HostTensor>(element::Type_t::f64, Shape{2, 2});
    HostTensorVector ins{make_host_tensor<element::Type_t::f32>(Shape{2, 3}),
                         make_host_tensor<element::Type_t::f64>(Shape{2, 2})};
    ASSERT_TRUE(fun->evaluate({result, result2}, ins));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_partial_shape(), (PartialShape{2, 3}));
    auto result_val = read_vector<float>(result);
    auto arg_val = read_vector<float>(ins[0]);
    ASSERT_EQ(result_val, arg_val);
    EXPECT_EQ(result2->get_element_type(), element::f64);
    EXPECT_EQ(result2->get_partial_shape(), (PartialShape{2, 2}));
    auto result_val2 = read_vector<double>(result2);
    auto arg_val2 = read_vector<double>(ins[1]);
    ASSERT_EQ(result_val2, arg_val2);
}

TEST(eval, evaluate_reshape_v1)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{2});
    auto dyn_reshape = make_shared<op::v1::Reshape>(data, pattern, false);
    auto func = make_shared<Function>(OutputVector{dyn_reshape}, ParameterVector{data, pattern});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(func->evaluate(
        {result_tensor},
        {make_host_tensor<element::Type_t::f32>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
         make_host_tensor<element::Type_t::i64>({2}, {5, 2})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{5, 2}));
    auto computed_val = read_vector<float>(result_tensor);
    vector<float> expected_val{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(computed_val, expected_val);
}

TEST(eval, evaluate_reshape_v1_negative_index)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{2});
    auto dyn_reshape = make_shared<op::v1::Reshape>(data, pattern, false);
    auto func = make_shared<Function>(OutputVector{dyn_reshape}, ParameterVector{data, pattern});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(func->evaluate(
        {result_tensor},
        {make_host_tensor<element::Type_t::f32>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
         make_host_tensor<element::Type_t::i64>({2}, {2, -1})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{2, 5}));
    auto computed_val = read_vector<float>(result_tensor);
    vector<float> expected_val{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(computed_val, expected_val);
}

TEST(eval, evaluate_reshape_v1_negative_index_zero_dim_zero_flag)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2, 2});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{6});
    auto dyn_reshape = make_shared<op::v1::Reshape>(data, pattern, true);
    auto func = make_shared<Function>(OutputVector{dyn_reshape}, ParameterVector{data, pattern});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        func->evaluate({result_tensor},
                       {make_host_tensor<element::Type_t::f32>(
                            {2, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                        make_host_tensor<element::Type_t::i64>({6}, {2, 0, 1, -1, 1, 2})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{2, 2, 1, 2, 1, 2}));
    auto computed_val = read_vector<float>(result_tensor);
    vector<float> expected_val{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    ASSERT_EQ(computed_val, expected_val);
}

TEST(eval, evaluate_reshape_v1_pattern_int16)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2, 2});
    auto pattern = make_shared<op::Parameter>(element::i16, Shape{6});
    auto dyn_reshape = make_shared<op::v1::Reshape>(data, pattern, true);
    auto func = make_shared<Function>(OutputVector{dyn_reshape}, ParameterVector{data, pattern});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        func->evaluate({result_tensor},
                       {make_host_tensor<element::Type_t::f32>(
                            {2, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                        make_host_tensor<element::Type_t::i16>({6}, {2, 0, 1, -1, 1, 2})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{2, 2, 1, 2, 1, 2}));
    auto computed_val = read_vector<float>(result_tensor);
    vector<float> expected_val{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    ASSERT_EQ(computed_val, expected_val);
}

TEST(eval, evaluate_convert)
{
    auto p = make_shared<op::Parameter>(element::f32, PartialShape{-1, -1});
    auto convert = make_shared<op::v0::Convert>(p, element::i64);
    auto fun = make_shared<Function>(OutputVector{convert}, ParameterVector{p});

    std::vector<std::vector<float>> inputs{{-1, 1}};
    std::vector<std::vector<int64_t>> expected_result{{-1, 1}};
    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate(
            {result}, {make_host_tensor<element::Type_t::f32>(Shape{1, 2}, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), (Shape{1, 2}));
        auto result_data = read_vector<int64_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}

TEST(eval, evaluate_abs)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto abs = make_shared<op::Abs>(p);
    auto fun = make_shared<Function>(OutputVector{abs}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_erf)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto erf = make_shared<op::Erf>(p);
    auto fun = make_shared<Function>(OutputVector{erf}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::erf(0.0f),
                        std::erf(-1.0f),
                        std::erf(-2.0f),
                        std::erf(-3.0f),
                        std::erf(4.0f),
                        std::erf(5.0f)};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_exp)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto exp = make_shared<op::Exp>(p);
    auto fun = make_shared<Function>(OutputVector{exp}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3}, {0.0f, -1.0f, -2.0f, -3.0f, 4.0f, 5.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::exp(0.0f),
                        std::exp(-1.0f),
                        std::exp(-2.0f),
                        std::exp(-3.0f),
                        std::exp(4.0f),
                        std::exp(5.0f)};
    ASSERT_FLOAT_VECTORS_EQ(expec, result_val);
}

TEST(eval, evaluate_floor)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto floor = make_shared<op::Floor>(p);
    auto fun = make_shared<Function>(OutputVector{floor}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{2, 2}, {-2.5f, -2.0f, 0.3f, 4.8f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-3.0f, -2.0f, 0.0f, 4.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_floor_int32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 2});
    auto floor = make_shared<op::Floor>(p);
    auto fun = make_shared<Function>(OutputVector{floor}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 2}, {-2, -136314888, 0x40000010, 0x40000001})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{-2, -136314888, 0x40000010, 0x40000001};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_log)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    auto log = make_shared<op::Log>(p);
    auto fun = make_shared<Function>(OutputVector{log}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result},
                      {make_host_tensor<element::Type_t::f32>(
                          Shape{2, 2, 2}, {0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{std::log(0.125f),
                        std::log(0.25f),
                        std::log(0.5f),
                        std::log(1.f),
                        std::log(2.f),
                        std::log(4.f),
                        std::log(8.f),
                        std::log(16.f)};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_negative_f32)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto negate = make_shared<op::Negative>(p);
    auto fun = make_shared<Function>(OutputVector{negate}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(
            Shape{2, 5},
            {1.35f, 8.76f, -8.0f, 17.234f, -2.121f, 1.0f, 8.7f, -8.92f, 17.0f, -1.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-1.35f, -8.76f, 8.0f, -17.234f, 2.121f, -1.0f, -8.7f, 8.92f, -17.0f, 1.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_negative_i32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    auto negate = make_shared<op::Negative>(p);
    auto fun = make_shared<Function>(OutputVector{negate}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -2, 1, 8, -8, 17, 0})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{-1, -8, 8, -17, 2, -1, -8, 8, -17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_relu_2Ffprop_f32)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 5});
    auto relu = make_shared<op::Relu>(p);
    auto fun = make_shared<Function>(OutputVector{relu}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -0.5, 0.1, 8.5, -8, 17, -0.5})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, 8, 0, 17, 0, 0.1, 8.5, 0, 17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_relu_2Ffprop_i32)
{
    auto p = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    auto relu = make_shared<op::Relu>(p);
    auto fun = make_shared<Function>(OutputVector{relu}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(
                                  Shape{2, 5}, {1, 8, -8, 17, -2, 1, 8, -8, 17, -1})}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    auto result_val = read_vector<int32_t>(result);
    vector<int32_t> expec{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_round)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{5});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1.0f, 2.0f, 2.0f, 2.0f, -4.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_round_2D)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    auto round = make_shared<op::v5::Round>(p, op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto fun = make_shared<Function>(OutputVector{round}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{3, 5},
                                                                      {0.1f,
                                                                       0.5f,
                                                                       0.9f,
                                                                       1.2f,
                                                                       1.5f,
                                                                       1.8f,
                                                                       2.3f,
                                                                       2.5f,
                                                                       2.7f,
                                                                       -1.1f,
                                                                       -1.5f,
                                                                       -1.9f,
                                                                       -2.2f,
                                                                       -2.5f,
                                                                       -2.8f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{
        0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sigmoid)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto sigmoid = make_shared<op::Sigmoid>(p);
    auto fun = make_shared<Function>(OutputVector{sigmoid}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    float x1 = 1.0f;
    float x2 = 4.0f;
    float sigma1 = 1.0f / (1.0f + std::exp(-x1));
    float sigma2 = 1.0f / (1.0f + std::exp(-x2));
    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(Shape{1, 1, 2, 2}, {x1, x2, x1, x2})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{sigma1, sigma2, sigma1, sigma2};
    EXPECT_EQ(result_val.size(), expec.size());
}

TEST(eval, evaluate_sign)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto sign = make_shared<op::Sign>(p);
    auto fun = make_shared<Function>(OutputVector{sign}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(Shape{2, 3}, {1, -2, 0, -4.8f, 4.8f, -0.0f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{1, -1, 0, -1, 1, 0};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_sin)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto sin = make_shared<op::Sin>(p);
    auto fun = make_shared<Function>(OutputVector{sin}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result},
        {make_host_tensor<element::Type_t::f32>(
            Shape{11}, {0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f})}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{0.00000000f,
                        0.24740396f,
                        -0.24740396f,
                        0.47942554f,
                        -0.47942554f,
                        0.84147098f,
                        -0.84147098f,
                        0.90929743f,
                        -0.90929743f,
                        -0.75680250f,
                        0.75680250f};
    ASSERT_FLOAT_VECTORS_EQ(expec, result_val);
}

TEST(eval, evaluate_sinh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sinh = make_shared<op::Sinh>(p);
    auto fun = make_shared<Function>(OutputVector{sinh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });
    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_sqrt)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto sqrt = make_shared<op::Sqrt>(p);
    auto fun = make_shared<Function>(OutputVector{sqrt}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{16, 4, 81, 100, 10000, 0};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{4, 2, 9, 10, 100, 0};
    ASSERT_FLOAT_VECTORS_EQ(expec, result_val);
}

TEST(eval, evaluate_acos)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto acos = make_shared<op::Acos>(p);
    auto fun = make_shared<Function>(OutputVector{acos}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::acos(x); });
    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_asin)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto asin = make_shared<op::Asin>(p);
    auto fun = make_shared<Function>(OutputVector{asin}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::asin(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_atan)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto atan = make_shared<op::Atan>(p);
    auto fun = make_shared<Function>(OutputVector{atan}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::atan(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_ceiling)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto ceil = make_shared<op::Ceiling>(p);
    auto fun = make_shared<Function>(OutputVector{ceil}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{-2.5f, -2.0f, 0.3f, 4.8f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{2, 2}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    vector<float> expec{-2.0f, -2.0f, 1.0f, 5.0f};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_cos)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto cos = make_shared<op::Cos>(p);
    auto fun = make_shared<Function>(OutputVector{cos}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::cos(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_cosh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto cosh = make_shared<op::Cosh>(p);
    auto fun = make_shared<Function>(OutputVector{cosh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::cosh(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_tan)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{11});
    auto tan = make_shared<op::Tan>(p);
    auto fun = make_shared<Function>(OutputVector{tan}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{11}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::tan(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_tanh)
{
    auto p = make_shared<op::Parameter>(element::f32, Shape{6});
    auto tanh = make_shared<op::Tanh>(p);
    auto fun = make_shared<Function>(OutputVector{tanh}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{6}, input)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    auto result_val = read_vector<float>(result);
    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return std::tanh(x); });

    ASSERT_FLOAT_VECTORS_EQ(input, result_val);
}

TEST(eval, evaluate_logical_not)
{
    auto p = make_shared<op::Parameter>(element::boolean, Shape{2, 2});
    auto logical_not = make_shared<op::v1::LogicalNot>(p);
    auto fun = make_shared<Function>(OutputVector{logical_not}, ParameterVector{p});
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(fun->evaluate(
        {result}, {make_host_tensor<element::Type_t::boolean>(Shape{2, 2}, {1, 0, 1, 0})}));
    EXPECT_EQ(result->get_element_type(), element::boolean);
    auto result_val = read_vector<char>(result);
    vector<char> expec{0, 1, 0, 1};
    ASSERT_EQ(result_val, expec);
}

TEST(eval, evaluate_dynamic_gather_v1)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto gather = make_shared<op::v1::Gather>(arg1, arg2, arg3);
    auto fun = make_shared<Function>(OutputVector{gather}, ParameterVector{arg1, arg2, arg3});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({3}, {1.0f, 2.0f, 3.0f}),
                               make_host_tensor<element::Type_t::i32>({2}, {1, 0}),
                               make_host_tensor<element::Type_t::i32>({1}, {0})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{2}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{2.0f, 1.0f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_gather_v1_scalar_axis)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto gather = make_shared<op::v1::Gather>(arg1, arg2, arg3);
    auto fun = make_shared<Function>(OutputVector{gather}, ParameterVector{arg1, arg2, arg3});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   {3, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
                               make_host_tensor<element::Type_t::i32>({1, 2}, {0, 2}),
                               make_host_tensor<element::Type_t::u64>({}, {1})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 1, 2}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1.0f, 1.2f, 2.0f, 2.2f, 3.0f, 3.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_gather_v7)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    int64_t batch_dims = 1;
    int32_t axis = 1;
    auto gather = make_shared<op::v7::Gather>(arg1, arg2, arg3, batch_dims);
    auto fun = make_shared<Function>(OutputVector{gather}, ParameterVector{arg1, arg2, arg3});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
                               make_host_tensor<element::Type_t::i32>({2, 2}, {1, 0, 1, 0}),
                               make_host_tensor<element::Type_t::i32>({1}, {axis})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{2, 2}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{2.0f, 1.0f, 5.0f, 4.0f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_gather_v7_axis_scalar)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    int64_t batch_dims = 0;
    int64_t axis = 1;
    auto gather = make_shared<op::v7::Gather>(arg1, arg2, arg3, batch_dims);
    auto fun = make_shared<Function>(OutputVector{gather}, ParameterVector{arg1, arg2, arg3});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                      {3, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f}),
                               make_host_tensor<element::Type_t::i32>({1, 2}, {0, 2}),
                               make_host_tensor<element::Type_t::i64>({}, {axis})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 1, 2}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1.0f, 1.2f, 2.0f, 2.2f, 3.0f, 3.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_concat)
{
    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto concat = make_shared<op::v0::Concat>(NodeVector{arg1, arg2}, 1);
    auto fun = make_shared<Function>(OutputVector{concat}, ParameterVector{arg1, arg2});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>({1, 1}, {1.0f}),
                               make_host_tensor<element::Type_t::f32>({1, 2}, {8.0f, 10.0f})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{1, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1.0f, 8.0f, 10.0f};
    ASSERT_EQ(cval, out);
}


TEST(eval, max_pool_v1_dynamic)
{
    Shape window_shape{3};
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto f = make_shared<Function>(
        make_shared<op::v1::MaxPool>(
            A, Strides(), Shape(), Shape(), window_shape, op::RoundingType::FLOOR),
        ParameterVector{A});
    auto result_tensor = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result_tensor},
                            {make_host_tensor<element::Type_t::f32>(
                                {1, 1, 14}, {0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{1, 1, 12}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0};
}

TEST(eval, evaluate_static_scatter_elements_update_basic)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};
    auto arg1 = make_shared<op::Parameter>(element::f32, data_shape);
    auto arg2 = make_shared<op::Parameter>(element::i32, indices_shape);
    auto arg3 = make_shared<op::Parameter>(element::f32, indices_shape);
    auto arg4 = make_shared<op::Parameter>(element::i64, Shape{});
    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result_tensor},
                      {make_host_tensor<element::Type_t::f32>(
                           data_shape, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}),
                       make_host_tensor<element::Type_t::i32>(indices_shape, {1, 0, 2, 0, 2, 1}),
                       make_host_tensor<element::Type_t::f32>(indices_shape,
                                                              {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                       make_host_tensor<element::Type_t::i64>({}, {0})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_shape(), (Shape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_elements_update_basic)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result_tensor},
                      {make_host_tensor<element::Type_t::f32>(
                           data_shape, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}),
                       make_host_tensor<element::Type_t::i32>(indices_shape, {1, 0, 2, 0, 2, 1}),
                       make_host_tensor<element::Type_t::f32>(indices_shape,
                                                              {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                       make_host_tensor<element::Type_t::i64>({}, {0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_elements_update_negative_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};
    const Shape axis_shape{};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result_tensor},
                      {make_host_tensor<element::Type_t::f32>(
                           data_shape, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}),
                       make_host_tensor<element::Type_t::i32>(indices_shape, {1, 0, 2, 0, 2, 1}),
                       make_host_tensor<element::Type_t::f32>(indices_shape,
                                                              {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                       make_host_tensor<element::Type_t::i64>(axis_shape, {-1})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{1.1f, 1.0f, 1.2f, 2.0f, 2.2f, 2.1f, 0.0f, 0.0f, 0.0f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_elements_update_1d_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{2, 3};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result_tensor},
                      {make_host_tensor<element::Type_t::f32>(
                           data_shape, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}),
                       make_host_tensor<element::Type_t::i32>(indices_shape, {1, 0, 2, 0, 2, 1}),
                       make_host_tensor<element::Type_t::f32>(indices_shape,
                                                              {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                       make_host_tensor<element::Type_t::i64>({1}, {0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{2.f, 1.1f, 0.0f, 1.f, 0.0f, 2.2f, 0.f, 2.1f, 1.2f};
    ASSERT_EQ(cval, out);
}

// Disabled test for disabled reference implementation
TEST(eval, DISABLED_evaluate_dynamic_scatter_elements_update_3d_i16)
{
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{2, 2, 3};

    auto arg1 = make_shared<op::Parameter>(element::i16, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i16, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i16, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::i16>(
                                   data_shape, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                               make_host_tensor<element::Type_t::i16>(
                                   indices_shape, {1, 0, 2, 0, 2, 1, 2, 2, 2, 0, 1, 0}),
                               make_host_tensor<element::Type_t::i16>(
                                   indices_shape, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
                               make_host_tensor<element::Type_t::i64>({}, {1})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::i16);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3, 3}));
    auto cval = read_vector<int16_t>(result_tensor);
    vector<int16_t> out{4, 2, 0, 1, 0, 6, 0, 5, 3, 10, 0, 12, 0, 11,
                        0, 7, 8, 9, 0, 0, 0, 0, 0, 0,  0, 0,  0};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_elements_update_one_elem_i32)
{
    const Shape data_shape{3, 3, 3};
    const Shape indices_shape{1, 1, 1};

    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_elements_update =
        make_shared<op::v3::ScatterElementsUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_elements_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::i32>(
                                   data_shape, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {1}),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {2}),
                               make_host_tensor<element::Type_t::i64>({}, {0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::i32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3, 3}));
    auto cval = read_vector<int32_t>(result_tensor);
    vector<int32_t> out{0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ASSERT_EQ(cval, out);
}

TEST(eval, topk_v1)
{
    Shape shape{2, 3, 2};
    Shape rshape{2, 2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    const auto k = op::Constant::create(element::i32, Shape{}, {2});
    auto B = make_shared<op::v1::TopK>(A, k, 1, "max", "index", element::i32);

    auto fun = make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                  Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);

    auto result1_val = read_vector<int32_t>(result1);

    vector<float> expec0{12, 9, 10, 4, 6, 3, 11, 7};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 0, 1, 2, 2};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v1_dyn)
{
    Shape shape{2, 3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v1::TopK>(A, k, 1, "max", "index", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {2})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 6, 3, 11, 7};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 0, 1, 2, 2};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v3_dyn)
{
    Shape shape{2, 3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v3::TopK>(A, k, 1, "max", "index", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {2})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 6, 3, 11, 7};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 0, 1, 2, 2};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v3_dyn_values)
{
    Shape shape{2, 3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v3::TopK>(A, k, 1, "max", "value", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {2})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 11, 7, 6, 3};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 2, 0, 1};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v3_dyn_values_k0)
{
    Shape shape{2, 3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v3::TopK>(A, k, 1, "max", "value", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {0})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 3, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 3, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 8, 2, 11, 7, 6, 3, 5, 1};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 0};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v1_dyn_k0)
{
    Shape shape{2, 3, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto k = make_shared<op::Parameter>(element::i64, Shape{});

    element::Type result_et{element::i32};
    auto B = make_shared<op::v1::TopK>(
        A, k, 1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, result_et);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i64>(Shape{}, {0})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 3, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 3, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);

    vector<float> expec0{12, 9, 10, 4, 8, 2, 11, 7, 6, 3, 5, 1};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 0};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v3_param_dyn_values_k0)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v3::TopK>(A, k, 1, "max", "value", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {0})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 3, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 3, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 8, 2, 11, 7, 6, 3, 5, 1};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 0};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v3_param_dyn_values_k2)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto k = make_shared<op::Parameter>(element::u32, Shape{});
    auto B = make_shared<op::v3::TopK>(A, k, 1, "max", "value", element::i32);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i32>(Shape{}, {2})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);
    vector<float> expec0{12, 9, 10, 4, 11, 7, 6, 3};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 2, 0, 1};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v1_param_dyn_k2)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto k = make_shared<op::Parameter>(element::i64, Shape{});
    auto axis = 1;

    element::Type result_et{element::i32};
    auto B = make_shared<op::v1::TopK>(
        A, k, axis, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, result_et);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i64>(Shape{}, {2})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 2, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 2, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);

    vector<float> expec0{12, 9, 10, 4, 11, 7, 6, 3};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 2, 0, 1};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, topk_v1_param_dyn_k0)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto k = make_shared<op::Parameter>(element::i64, Shape{});

    element::Type result_et{element::i32};

    auto B = make_shared<op::v1::TopK>(
        A, k, 1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES, result_et);

    auto fun =
        make_shared<Function>(OutputVector{B->output(0), B->output(1)}, ParameterVector{A, k});

    auto result0 = make_shared<HostTensor>();
    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result0, result1},
                              {make_host_tensor<element::Type_t::f32>(
                                   Shape{2, 3, 2}, {12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7}),
                               make_host_tensor<element::Type_t::i64>(Shape{}, {0})}));
    EXPECT_EQ(result0->get_element_type(), element::f32);
    EXPECT_EQ(result0->get_partial_shape(), (PartialShape{2, 3, 2}));
    EXPECT_EQ(result1->get_element_type(), element::i32);
    EXPECT_EQ(result1->get_partial_shape(), (PartialShape{2, 3, 2}));
    auto result0_val = read_vector<float>(result0);
    auto result1_val = read_vector<int32_t>(result1);

    vector<float> expec0{12, 9, 10, 4, 8, 2, 11, 7, 6, 3, 5, 1};
    ASSERT_EQ(result0_val, expec0);

    vector<int32_t> expec1{0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 0};
    ASSERT_EQ(result1_val, expec1);
}

TEST(eval, reduce_logical_and__neg_axis)
{
    const auto data = make_shared<op::Parameter>(element::boolean, Shape{2, 2, 2});
    const auto axes = make_shared<op::Parameter>(element::i64, Shape{});

    const auto op = make_shared<op::v1::ReduceLogicalAnd>(data, axes);

    auto fun = make_shared<Function>(op, ParameterVector{data, axes});

    auto result = make_shared<HostTensor>();

    // when ReduceLogicalAnd node evaluator returns false -> the Function object throws
    EXPECT_THROW(
        fun->evaluate({result},
                      {
                          make_host_tensor<element::Type_t::boolean>(
                              Shape{2, 2, 2}, {true, false, true, false, true, false, true, false}),
                          make_host_tensor<element::Type_t::i64>(Shape{}, {-1}),
                      }),
        ngraph::ngraph_error);
}

TEST(eval, evaluate_static_scatter_update_basic_axes_indices_i32)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{1, 2};
    const Shape updates_shape{1, 2, 3};

    auto arg1 = make_shared<op::Parameter>(element::f32, data_shape);
    auto arg2 = make_shared<op::Parameter>(element::i32, indices_shape);
    auto arg3 = make_shared<op::Parameter>(element::f32, updates_shape);
    auto arg4 = make_shared<op::Parameter>(element::i32, Shape{});
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   data_shape, std::vector<float>(shape_size(data_shape))),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {1, 2}),
                               make_host_tensor<element::Type_t::f32>(
                                   updates_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                               make_host_tensor<element::Type_t::i32>({}, {0})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_shape(), (Shape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{0.f, 0.f, 0.f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_static_scatter_update_basic_axes_indices_i64)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{1, 2};
    const Shape updates_shape{1, 2, 3};

    auto arg1 = make_shared<op::Parameter>(element::f32, data_shape);
    auto arg2 = make_shared<op::Parameter>(element::i64, indices_shape);
    auto arg3 = make_shared<op::Parameter>(element::f32, updates_shape);
    auto arg4 = make_shared<op::Parameter>(element::i64, Shape{});
    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   data_shape, std::vector<float>(shape_size(data_shape))),
                               make_host_tensor<element::Type_t::i64>(indices_shape, {1, 2}),
                               make_host_tensor<element::Type_t::f32>(
                                   updates_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                               make_host_tensor<element::Type_t::i64>({}, {0})}));
    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_shape(), (Shape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{0.f, 0.f, 0.f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_update_basic)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{1, 2};
    const Shape updates_shape{1, 2, 3};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   data_shape, std::vector<float>(shape_size(data_shape))),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {1, 2}),
                               make_host_tensor<element::Type_t::f32>(
                                   updates_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                               make_host_tensor<element::Type_t::i64>({}, {0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{0.f, 0.f, 0.f, 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_update_negative_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{1, 2};
    const Shape updates_shape{3, 1, 2};
    const Shape axis_shape{};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   data_shape, std::vector<float>(shape_size(data_shape))),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {1, 2}),
                               make_host_tensor<element::Type_t::f32>(
                                   updates_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                               make_host_tensor<element::Type_t::i64>(axis_shape, {-1})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{0.f, 1.0f, 1.1f, 0.0f, 1.2f, 2.0f, 0.0f, 2.1f, 2.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_update_1d_axis)
{
    const Shape data_shape{3, 3};
    const Shape indices_shape{1, 2};
    const Shape updates_shape{3, 1, 2};

    auto arg1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result_tensor},
                              {make_host_tensor<element::Type_t::f32>(
                                   data_shape, std::vector<float>(shape_size(data_shape))),
                               make_host_tensor<element::Type_t::i32>(indices_shape, {1, 2}),
                               make_host_tensor<element::Type_t::f32>(
                                   updates_shape, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f}),
                               make_host_tensor<element::Type_t::i64>({1}, {1})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::f32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3}));
    auto cval = read_vector<float>(result_tensor);
    vector<float> out{0.f, 1.0f, 1.1f, 0.0f, 1.2f, 2.0f, 0.0f, 2.1f, 2.2f};
    ASSERT_EQ(cval, out);
}

TEST(eval, evaluate_dynamic_scatter_update_one_elem_i32)
{
    const Shape data_shape{3, 3, 2};
    const Shape indices_shape{1, 1};
    const Shape updates_shape{1, 1, 3, 2};

    auto arg1 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg2 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg3 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto arg4 = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto scatter_update = make_shared<op::v3::ScatterUpdate>(arg1, arg2, arg3, arg4);
    auto fun = make_shared<Function>(OutputVector{scatter_update},
                                     ParameterVector{arg1, arg2, arg3, arg4});
    auto result_tensor = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result_tensor},
                      {make_host_tensor<element::Type_t::i32>(
                           data_shape, std::vector<int32_t>(shape_size(data_shape))),
                       make_host_tensor<element::Type_t::i32>(indices_shape, {1}),
                       make_host_tensor<element::Type_t::i32>(updates_shape, {1, 2, 3, 4, 5, 6}),
                       make_host_tensor<element::Type_t::i64>({}, {0})}));

    EXPECT_EQ(result_tensor->get_element_type(), element::i32);
    EXPECT_EQ(result_tensor->get_partial_shape(), (PartialShape{3, 3, 2}));
    auto cval = read_vector<int32_t>(result_tensor);
    vector<int32_t> out{0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0};
    ASSERT_EQ(cval, out);
}
