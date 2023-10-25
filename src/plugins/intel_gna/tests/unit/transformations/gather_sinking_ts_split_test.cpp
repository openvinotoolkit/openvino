// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/ts_split_backward.hpp"

using namespace ov;
using namespace ov::opset12;

namespace {
std::vector<size_t> TSSplit_Backward_indexes(size_t size, size_t initial_value) {
    return std::vector<size_t>{0, 2, 4, 6, 1, 3, 5, 7};
}
}  // namespace

TEST(TSSplit, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 4, 1, 2});
        auto split_axis = Constant::create(element::i64, ov::Shape{}, ov::Shape{1});
        auto split = std::make_shared<Split>(input_params, split_axis, 1);
        auto transpose_const = Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});
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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 4, 1, 2});
        auto reshape_const1 = Constant::create(element::i64, ov::Shape{2}, ov::Shape{1, 8});
        auto reshape1 = std::make_shared<Reshape>(input_params, reshape_const1, false);
        auto gather = make_gather(reshape1, TSSplit_Backward_indexes, /* axis */ 1);
        auto split_axis = Constant::create(element::i64, ov::Shape{}, ov::Shape{1});
        auto split_lengths = Constant::create(element::i64, ov::Shape{1}, ov::Shape{8});
        auto split = std::make_shared<VariadicSplit>(gather, split_axis, split_lengths);
        auto reshape_const2 = Constant::create(element::i64, ov::Shape{4}, ov::Shape{1, 1, 2, 4});
        auto reshape2 = std::make_shared<Reshape>(split->output(0), reshape_const2, false);
        const auto result = std::make_shared<Result>(reshape2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
