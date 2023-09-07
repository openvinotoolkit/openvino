// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking.hpp"

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <ops/gna_convolution.hpp>
#include <ops/gna_max_pool.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "transformations/gather_sinking_matmul.hpp"
#include "transformations/gather_sinking_reshape.hpp"
#include "transformations/gather_sinking_split.hpp"
#include "transformations/replace_gna_nhwc_layers.hpp"

using namespace ov;
using namespace ov::opset10;

TEST(TransposeNCHW, Convolution) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 41, 1});
        auto kernel = Constant::create(ov::element::f32, {4, 1, 3, 1}, {1});

        auto convolution = std::make_shared<Convolution>(input_params,
                                                         kernel,
                                                         Strides{2, 1},
                                                         CoordinateDiff{0, 0},
                                                         CoordinateDiff{0, 0},
                                                         Strides{1, 1});

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

        auto transpose_before_const = Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});

        auto transpose_before = std::make_shared<Transpose>(input_params, transpose_before_const);

        auto kernel = Constant::create(ov::element::f32, {4, 1, 3, 1}, {1});

        auto transpose_conv_const = Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});

        auto transpose_conv_before = std::make_shared<Transpose>(input_params, transpose_conv_const);

        auto transpose_conv_constant = std::make_shared<Transpose>(kernel, transpose_conv_const);

        auto convolution = std::make_shared<ov::intel_gna::op::GNAConvolution>(transpose_before,
                                                                               transpose_conv_constant,
                                                                               Strides{2, 1},
                                                                               CoordinateDiff{0, 0},
                                                                               CoordinateDiff{0, 0},
                                                                               Strides{1, 1});

        auto transpose_after_const = Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});

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

        auto max_pool =
            std::make_shared<ov::op::v1::MaxPool>(input_params, Strides{2, 1}, Shape{0, 0}, Shape{0, 0}, Shape{4, 1});

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

        auto transpose_before_const = Constant::create(element::i32, Shape{4}, {0, 2, 3, 1});

        auto transpose_before = std::make_shared<Transpose>(input_params, transpose_before_const);

        auto max_pool = std::make_shared<ov::intel_gna::op::GNAMaxPool>(transpose_before,
                                                                        Strides{2, 1},
                                                                        Shape{0, 0},
                                                                        Shape{0, 0},
                                                                        Shape{4, 1});

        auto transpose_after_const = Constant::create(element::i32, Shape{4}, {0, 3, 1, 2});

        auto transpose_after = std::make_shared<Transpose>(max_pool, transpose_after_const);

        const auto result = std::make_shared<Result>(transpose_after);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
