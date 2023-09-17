// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

namespace testing {

TEST(GatherSinkingTransposeReshape, ForwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 4});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(tanh0, transpose_order);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 4});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        std::vector<size_t> gather_indices = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
        auto gather_indices_const =
            std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(reshape, gather_indices_const, gather_axis_const);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST(GatherSinkingTransposeReshape, ForwardSinking3D) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 14, 4});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(tanh0, transpose_order);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, 56});
        auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeForward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 14, 4});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto generate_indices = []() -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 14; ++i) {
                    indices.push_back(j + 4 * i);
                }
            }
            return indices;
        };
        auto gather_indices = generate_indices();
        auto gather_indices_const =
            std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(reshape, gather_indices_const, gather_axis_const);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingTransposeReshape, BackwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 2, 2});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 2, 1, 2});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(reshape, transpose_order);

        auto tanh1 = std::make_shared<Tanh>(transpose);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingTransposeReshapeBackward>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 1, 2, 2});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const1 = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 2, 1, 2});
        auto reshape1 = std::make_shared<Reshape>(tanh0, reshape_const1, false);

        auto reshape_const2 = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 1, 2, 2});
        auto reshape2 = std::make_shared<Reshape>(reshape1, reshape_const2, false);

        auto tanh1 = std::make_shared<Tanh>(reshape2);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

}  // namespace testing
