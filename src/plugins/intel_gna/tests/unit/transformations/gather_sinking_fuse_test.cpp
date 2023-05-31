// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

#include <ngraph/function.hpp>
#include <openvino/opsets/opset10.hpp>
#include <ops/gna_convolution.hpp>
#include <ops/gna_max_pool.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "transformations/transpose_nchw.hpp"
#include "transformations/gather_sinking_matmul.hpp"
#include "transformations/gather_sinking_split.hpp"
#include "transformations/gather_sinking_reshape.hpp"
#include "transformations/gather_sinking.hpp"
#include "transformations/ts_concat.hpp"
#include "transformations/ts_split.hpp"

#include "ngraph/pass/visualize_tree.hpp"  // DEBUG

using namespace ov;
using namespace ov::opset10;

TEST(TransposeNCHW, Convolution) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 41, 1});
        auto kernel = Constant::create(ov::element::f32, {4,1,3,1}, {1});

        auto convolution = std::make_shared<Convolution>(input_params,
                                                         kernel,
                                                         ngraph::Strides{2, 1},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::Strides{1, 1});

        const auto result = std::make_shared<Result>(convolution);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAConvolution>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 41, 1});

        auto transpose_before_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose_before = std::make_shared<Transpose>(input_params, transpose_before_const);

        auto kernel = Constant::create(ov::element::f32, {4,1,3,1}, {1});

        auto transpose_conv_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose_conv_before = std::make_shared<Transpose>(input_params, transpose_conv_const);

        auto transpose_conv_constant = std::make_shared<Transpose>(kernel, transpose_conv_const);

        auto convolution = std::make_shared<ov::intel_gna::op::GNAConvolution>(transpose_before,
                                                         transpose_conv_constant,
                                                         ngraph::Strides{2, 1},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::Strides{1, 1});

        auto transpose_after_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,3,1,2});

        auto transpose_after = std::make_shared<Transpose>(convolution, transpose_after_const);

        const auto result = std::make_shared<Result>(transpose_after);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransposeNCHW, MaxPool) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 41, 1});

        auto max_pool = std::make_shared<ov::op::v1::MaxPool>(input_params,
                                                         ngraph::Strides{2, 1},
                                                         Shape{0, 0},
                                                         Shape{0, 0},
                                                         Shape{4, 1});

        const auto result = std::make_shared<Result>(max_pool);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::SubstituteGNAMaxPool>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 41, 1});

        auto transpose_before_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose_before = std::make_shared<Transpose>(input_params, transpose_before_const);

        auto max_pool = std::make_shared<ov::intel_gna::op::GNAMaxPool>(transpose_before,
                                                         ngraph::Strides{2, 1},
                                                         Shape{0, 0},
                                                         Shape{0, 0},
                                                         Shape{4, 1});

        auto transpose_after_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,3,1,2});

        auto transpose_after = std::make_shared<Transpose>(max_pool, transpose_after_const);

        const auto result = std::make_shared<Result>(transpose_after);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

void ShiftLeft(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.begin(), vec.begin() + k, buffer.begin());

    for (int i = k; i < vec.size(); ++i) {
        vec[i - k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.end() - k);
}

void ShiftRight(std::vector<size_t>& vec, size_t k) {
    if (k > vec.size())
        return;
    std::vector<size_t> buffer(k);
    std::copy(vec.end() - k, vec.end(), buffer.begin());

    for (int i = vec.size() - 1 - k; i >= 0; --i) {
        vec[i + k] = vec[i];
    }

    std::copy(buffer.begin(), buffer.end(), vec.begin());
}

std::vector<size_t> GatherForward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    ShiftLeft(vec, 2);
    return vec;
}

std::vector<size_t> GatherBackward(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value); // Not the same as in binary tests
    ShiftRight(vec, 2);
    return vec;
}

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;
using Output = ov::Output<ov::Node>;

template <typename CreateIndicesF>
std::shared_ptr<Gather> MakeGather(NodePtr input_node, CreateIndicesF create_indices_func, size_t axis) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    const std::vector<size_t> indexes = create_indices_func(input_shape[axis], 0);

    auto gather_indexes_node = Constant::create(ngraph::element::i64, ov::Shape{indexes.size()}, indexes);

    auto gather_axis_node = Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});

    return std::make_shared<Gather>(input_node->output(0), gather_indexes_node, gather_axis_node);
}

TEST(GatherSinkingMatMul, Forward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        
        auto gather = MakeGather(input_params, GatherForward, /* axis */ 1);

        auto input_const1 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul1 = std::make_shared<MatMul>(gather, input_const1);

        auto input_const2 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingMatmulForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        auto input_const = Constant::create(ov::element::f32, {20,20}, {1});

        auto gather = MakeGather(input_const, GatherBackward, /* axis */ 0);

        auto input_const1 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, gather);

        auto input_const2 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingMatMul, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});

        auto input_const1 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, input_const1);

        auto input_const2 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        auto gather = MakeGather(matmul2, GatherForward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingMatmulBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        auto input_const = Constant::create(ov::element::f32, {20,20}, {1});

        auto gather = MakeGather(input_const, GatherForward, /* axis */ 1);

        auto input_const1 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul1 = std::make_shared<MatMul>(input_params, gather);

        auto input_const2 = Constant::create(ov::element::f32, {20,20}, {1});
        auto matmul2 = std::make_shared<MatMul>(input_const2, matmul1);

        const auto result = std::make_shared<Result>(matmul2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingSplit, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});

        auto split_axis1 = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{0});
        auto split1 = std::make_shared<Split>(input_params, split_axis1, 2);

        auto split_axis2 = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{0});
        auto split2 = std::make_shared<Split>(split1, split_axis2, 2);

        auto gather = MakeGather(split2, GatherForward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingSplitBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});

        auto gather = MakeGather(input_params, GatherForward, /* axis */ 1);

        auto split_axis1 = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{0});
        auto split1 = std::make_shared<Split>(gather, split_axis1, 2);

        auto split_axis2 = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{0});
        auto split2 = std::make_shared<Split>(split1, split_axis2, 2);

        const auto result = std::make_shared<Result>(split2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingReshape, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1,168});

        auto reshape_const1 = Constant::create(ngraph::element::i64, ov::Shape{4}, ov::Shape{1,168,1,1});
        auto reshape1 = std::make_shared<Reshape>(input_params, reshape_const1, false);

        auto reshape_const2 = Constant::create(ngraph::element::i64, ov::Shape{5}, ov::Shape{1,168,1,1,1});
        auto reshape2 = std::make_shared<Reshape>(reshape1, reshape_const2, false);

        auto gather = MakeGather(reshape2, GatherForward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingReshapeBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1,168});

        auto gather = MakeGather(input_params, GatherForward, /* axis */ 1);

        auto reshape_const1 = Constant::create(ngraph::element::i64, ov::Shape{4}, ov::Shape{1,168,1,1});
        auto reshape1 = std::make_shared<Reshape>(gather, reshape_const1, false);

        auto reshape_const2 = Constant::create(ngraph::element::i64, ov::Shape{5}, ov::Shape{1,168,1,1,1});
        auto reshape2 = std::make_shared<Reshape>(reshape1, reshape_const2, false);

        const auto result = std::make_shared<Result>(reshape2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingGeneral, General) {
    std::shared_ptr<Model> function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        
        auto gather = MakeGather(input_params1, GatherForward, /* axis */ 1);

        auto tanh = std::make_shared<Tanh>(input_params2);
        auto mult = std::make_shared<Multiply>(gather, tanh);
        auto sinh = std::make_shared<Sinh>(mult);

        const auto result = std::make_shared<Result>(sinh);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingGeneral>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20,20});
        
        auto gather1 = MakeGather(input_params2, GatherBackward, /* axis */ 1);

        auto tanh = std::make_shared<Tanh>(gather1);
        auto mult = std::make_shared<Multiply>(input_params1, tanh);
        auto sinh = std::make_shared<Sinh>(mult);

        auto gather2 = MakeGather(sinh, GatherForward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

#if 0
std::vector<size_t> TSConcat_Forward_indexes(size_t size, size_t initial_value) {
    return std::vector<size_t>{0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
}

TEST(TSConcat, Forward) {
    std::shared_ptr<Model> function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});

        auto transpose_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose = std::make_shared<Transpose>(input_params1, transpose_const);

        auto concat = std::make_shared<Concat>(NodeVector{transpose, input_params2}, 0);

        const auto result = std::make_shared<Result>(concat);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::TSConcatForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{2,2,2,2});

        auto reshape_const1 = Constant::create(ngraph::element::i64, ov::Shape{2}, ov::Shape{1,16});
        auto reshape1 = std::make_shared<Reshape>(input_params1, reshape_const1, false);

        auto reshape_const2 = Constant::create(ngraph::element::i64, ov::Shape{2}, ov::Shape{1,16});
        auto reshape2 = std::make_shared<Reshape>(input_params2, reshape_const2, false);

        auto concat = std::make_shared<Concat>(NodeVector{reshape1, reshape2}, 1);

        auto gather = MakeGather(concat, TSConcat_Forward_indexes, /* axis */ 1);

        auto reshape_const3 = Constant::create(ngraph::element::i64, ov::Shape{4}, ov::Shape{4,2,2,2});
        auto reshape3 = std::make_shared<Reshape>(gather, reshape_const3, false);

        const auto result = std::make_shared<Result>(reshape3);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
#endif

std::vector<size_t> TSSplit_Backward_indexes(size_t size, size_t initial_value) {
    return std::vector<size_t>{0, 2, 4, 6, 1, 3, 5, 7};
}

TEST(TSSplit, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1,4,1,2});

        auto split_axis = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{1});
        auto split = std::make_shared<Split>(input_params, split_axis, 1);

        auto transpose_const = Constant::create(ngraph::element::i64,
                                                            ngraph::Shape{4},
                                                            {0,2,3,1});

        auto transpose = std::make_shared<Transpose>(split->output(0), transpose_const);

        const auto result = std::make_shared<Result>(transpose);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::TSSplitBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1,4,1,2});

        auto reshape_const1 = Constant::create(ngraph::element::i64, ov::Shape{2}, ov::Shape{1,8});
        auto reshape1 = std::make_shared<Reshape>(input_params, reshape_const1, false);

        auto gather = MakeGather(reshape1, TSSplit_Backward_indexes, /* axis */ 1);

        auto split_axis = Constant::create(ngraph::element::i64, ov::Shape{}, ov::Shape{1});
        auto split = std::make_shared<Split>(gather, split_axis, 1);

        auto reshape_const2 = Constant::create(ngraph::element::i64, ov::Shape{4}, ov::Shape{1, 1, 2, 4});
        auto reshape2 = std::make_shared<Reshape>(split->output(0), reshape_const2, false);

        const auto result = std::make_shared<Result>(reshape2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
