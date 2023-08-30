// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/gather_sinking.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

TEST(GatherSinkingGeneral, General) {
    std::shared_ptr<Model> function;
    {
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather = make_gather(input_params1, gather_forward, /* axis */ 1);

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
        auto input_params1 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});
        auto input_params2 = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather1 = make_gather(input_params2, gather_backward, /* axis */ 1);

        auto tanh = std::make_shared<Tanh>(gather1);
        auto mult = std::make_shared<Multiply>(input_params1, tanh);
        auto sinh = std::make_shared<Sinh>(mult);

        auto gather2 = make_gather(sinh, gather_forward, /* axis */ 1);

        const auto result = std::make_shared<Result>(gather2);
        reference_function =
            std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params1, input_params2});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
