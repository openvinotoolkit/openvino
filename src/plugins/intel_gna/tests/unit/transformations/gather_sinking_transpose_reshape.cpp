// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/gather_sinking_transpose_reshape.hpp"

#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

#include "ngraph/pass/visualize_tree.hpp" // DEBUG

using namespace ov;
using namespace ov::opset9;

namespace testing {

TEST(GatherSinkingTransposeReshape, ForwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, -1});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto generate_indices = []() -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (int i = 0; i < 80; ++i) {
                indices.push_back(i);
                indices.push_back(i + 80);
                indices.push_back(i + 160);
            }
            return indices;
        };
        auto gather_indices = generate_indices();
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(reshape, gather_indices_const, gather_axis_const);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
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

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int>{1, 56});
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
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(reshape, gather_indices_const, gather_axis_const);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}


TEST(GatherSinkingTransposeReshape, BackwardSinking) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 240});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int>{1, 3, 80});
        auto reshape = std::make_shared<Reshape>(tanh0, reshape_const, false);

        auto transpose_order = std::make_shared<Constant>(element::u64, Shape{3}, Shape{0, 2, 1});
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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 240});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto generate_indices = []() -> std::vector<int64_t> {
            std::vector<int64_t> indices;
            for (int i = 0; i < 80; ++i) {
                indices.push_back(i);
                indices.push_back(i + 80);
                indices.push_back(i + 160);
            }
            return indices;
        };
        auto gather_indices = generate_indices();
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(tanh0, gather_indices_const, gather_axis_const);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int>{1, 80, 3});
        auto reshape = std::make_shared<Reshape>(gather, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingTransposeReshape, BackwardSinking3D) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 56});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 4, 1, 14});
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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 56});
        auto tanh0 = std::make_shared<Tanh>(input_params);

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
        auto gather_indices_const = std::make_shared<Constant>(element::i64, Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto gather = std::make_shared<Gather>(tanh0, gather_indices_const, gather_axis_const);

        auto reshape_const = std::make_shared<Constant>(element::i64, Shape{4}, std::vector<int>{1, 1, 14, 4});
        auto reshape = std::make_shared<Reshape>(gather, reshape_const, false);

        auto tanh1 = std::make_shared<Tanh>(reshape);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

} // namespace testing
