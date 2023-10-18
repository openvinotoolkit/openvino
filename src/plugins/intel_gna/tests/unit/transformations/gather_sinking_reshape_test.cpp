// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_reshape.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;
using Output = ov::Output<ov::Node>;

TEST(GatherSinkingReshape, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 168});

        auto reshape_const1 = Constant::create(element::i64, ov::Shape{4}, ov::Shape{1, 168, 1, 1});
        auto reshape1 = std::make_shared<Reshape>(input_params, reshape_const1, false);

        auto reshape_const2 = Constant::create(element::i64, ov::Shape{5}, ov::Shape{1, 168, 1, 1, 1});
        auto reshape2 = std::make_shared<Reshape>(reshape1, reshape_const2, false);

        auto gather = make_gather(reshape2, gather_forward, /* axis */ 1);

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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{1, 168});

        auto gather = make_gather(input_params, gather_forward, /* axis */ 1);

        auto reshape_const1 = Constant::create(element::i64, ov::Shape{4}, ov::Shape{1, 168, 1, 1});
        auto reshape1 = std::make_shared<Reshape>(gather, reshape_const1, false);

        auto reshape_const2 = Constant::create(element::i64, ov::Shape{5}, ov::Shape{1, 168, 1, 1, 1});
        auto reshape2 = std::make_shared<Reshape>(reshape1, reshape_const2, false);

        const auto result = std::make_shared<Result>(reshape2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
