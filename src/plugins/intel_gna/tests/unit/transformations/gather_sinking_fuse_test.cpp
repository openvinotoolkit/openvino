// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_fuse.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;

namespace {

std::shared_ptr<Gather> make_gather(NodePtr input_node, const std::vector<size_t>& indices, size_t axis) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    auto gather_indexes_node = Constant::create(element::i64, ov::Shape{indices.size()}, indices);

    auto gather_axis_node = Constant::create(element::i64, Shape{}, {axis});

    return std::make_shared<Gather>(input_node, gather_indexes_node, gather_axis_node);
}

}  // namespace

TEST(GatherSinkingFuse, Remove) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto input_gather = make_gather(tanh0, std::vector<size_t>{2, 0, 1}, /* axis */ 1);
        auto output_gather = make_gather(input_gather, std::vector<size_t>{1, 2, 0}, /* axis */ 1);

        auto tanh1 = std::make_shared<Tanh>(output_gather);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingFuse>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto tanh1 = std::make_shared<Tanh>(tanh0);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingFuse, DifferentAxis) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto input_gather = make_gather(tanh0, std::vector<size_t>{2, 0, 1}, /* axis */ 1);
        auto output_gather = make_gather(input_gather, std::vector<size_t>{1, 2, 0}, /* axis */ 2);

        auto tanh1 = std::make_shared<Tanh>(output_gather);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingFuse>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function = function->clone();

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(GatherSinkingFuse, Combine) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto input_gather = make_gather(tanh0, std::vector<size_t>{2, 0, 1}, /* axis */ 1);
        auto output_gather = make_gather(input_gather, std::vector<size_t>{1, 0, 2}, /* axis */ 1);

        auto tanh1 = std::make_shared<Tanh>(output_gather);
        const auto result = std::make_shared<Result>(tanh1);
        function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    std::shared_ptr<Model> orig_function = function->clone();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::GatherSinkingFuse>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    std::shared_ptr<Model> reference_function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 3, 80});
        auto tanh0 = std::make_shared<Tanh>(input_params);

        auto gather = make_gather(tanh0, std::vector<size_t>{0, 2, 1}, /* axis */ 1);

        auto tanh1 = std::make_shared<Tanh>(gather);
        const auto result = std::make_shared<Result>(tanh1);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}