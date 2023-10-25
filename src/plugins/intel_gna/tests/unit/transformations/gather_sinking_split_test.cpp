// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_split.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

TEST(GatherSinkingSplit, Backward) {
    std::shared_ptr<Model> function;
    {
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto split_axis1 = Constant::create(element::i64, ov::Shape{}, ov::Shape{0});
        auto split1 = std::make_shared<Split>(input_params, split_axis1, 2);

        auto split_axis2 = Constant::create(element::i64, ov::Shape{}, ov::Shape{0});
        auto split2 = std::make_shared<Split>(split1, split_axis2, 2);

        auto gather = make_gather(split2, gather_forward, /* axis */ 1);

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
        auto input_params = std::make_shared<Parameter>(element::Type_t::f32, Shape{20, 20});

        auto gather = make_gather(input_params, gather_forward, /* axis */ 1);

        auto split_axis1 = Constant::create(element::i64, ov::Shape{}, ov::Shape{0});
        auto split1 = std::make_shared<Split>(gather, split_axis1, 2);

        auto split_axis2 = Constant::create(element::i64, ov::Shape{}, ov::Shape{0});
        auto split2 = std::make_shared<Split>(split1, split_axis2, 2);

        const auto result = std::make_shared<Result>(split2);
        reference_function = std::make_shared<Model>(OutputVector{result}, ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}
