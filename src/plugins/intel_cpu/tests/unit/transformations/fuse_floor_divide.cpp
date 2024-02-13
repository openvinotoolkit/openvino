// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "openvino/op/parameter.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/convert_precision.hpp"
#include <transformations/cpu_opset/common/pass/fuse_floor_divide.hpp>


#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace std;
using namespace testing;
using namespace ov;
using namespace ov::op;



TEST(TransformationTests, fuse_floor_divide_1) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{1, 100});
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v1::Less>(x, zero_const);
        auto y_less_cond = make_shared<v1::Less>(y, zero_const);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x, y, false);

        auto mod_xy = make_shared<v1::Mod>(x, y);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{x, y});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{1, 100});
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});
        auto div = make_shared<v1::Divide>(x, y, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{x, y});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}


TEST(TransformationTests, fuse_floor_divide_2) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{1, 100});
        auto y_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v1::Less>(x, zero_const);
        auto y_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x, y_const, false);

        auto mod_xy = make_shared<v1::Mod>(x, y_const);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{x});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{1, 100});
        auto y_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);
        auto div = make_shared<v1::Divide>(x, y_const, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{x});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, fuse_floor_divide_2_) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{2, 4, 5});
        auto y_const = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v1::Less>(x, zero_const);
        auto y_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x, y_const, false);

        auto mod_xy = make_shared<v1::Mod>(x, y_const);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{x});
        precisions_map map = {
            {ov::element::boolean, ov::element::u8},
        };

        manager.register_pass<ov::pass::ConvertPrecision>(map);
        // manager.register_pass<ov::pass::VisualizeTree>("before.svg");
        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        // manager.register_pass<ov::pass::VisualizeTree>("after.svg");
        manager.run_passes(model);
    }

    {
        auto x = make_shared<v0::Parameter>(element::i32, Shape{2, 4, 5});
        auto y_const = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto div = make_shared<v1::Divide>(x, y_const, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{x});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, fuse_floor_divide_3) {
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto y_less_cond = make_shared<v1::Less>(y, zero_const);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x_const, y, false);

        auto mod_xy = make_shared<v1::Mod>(x_const, y);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{y});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});
        auto div = make_shared<v1::Divide>(x_const, y, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{y});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST(TransformationTests, fuse_floor_divide_4) {
    // negative test when constants are different
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto non_zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 42);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto y_less_cond = make_shared<v1::Less>(y, non_zero_const);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x_const, y, false);

        auto mod_xy = make_shared<v1::Mod>(x_const, y);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{y});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});
        auto div = make_shared<v1::Divide>(x_const, y, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{y});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_FALSE(result.valid) << result.message;
}

TEST(TransformationTests, fuse_floor_divide_5) {
    // negative test when constants are different
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        // to match it should've been -1
        auto wrong_const = make_shared<v0::Constant>(element::i32, Shape{}, 42);

        auto x_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto y_less_cond = make_shared<v1::Less>(y, zero_const);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x_const, y, false);

        auto mod_xy = make_shared<v1::Mod>(x_const, y);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, wrong_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{y});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});
        auto div = make_shared<v1::Divide>(x_const, y, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{y});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_FALSE(result.valid) << result.message;
}

TEST(TransformationTests, fuse_floor_divide_6) {
    // negative test: both Less conditions are Const, it is not FloorDiv subgraph, we shuold not fuse it
    shared_ptr<Model> model, model_ref;
    pass::Manager manager;
    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});

        auto zero_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto minus_one_const = make_shared<v0::Constant>(element::i32, Shape{}, -1);

        auto x_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto y_less_cond = make_shared<v0::Constant>(element::boolean, Shape{}, 0);
        auto xor_cond = make_shared<v1::LogicalXor>(x_less_cond, y_less_cond);

        auto div = make_shared<v1::Divide>(x_const, y, false);

        auto mod_xy = make_shared<v1::Mod>(x_const, y);
        auto cond_mod = make_shared<v1::NotEqual>(mod_xy, zero_const);

        auto cond = make_shared<v1::LogicalAnd>(cond_mod, xor_cond);
        auto reminder = make_shared<v1::Select>(cond, minus_one_const, zero_const);
        auto trunc_div = make_shared<v1::Add>(div, reminder);

        model = make_shared<Model>(NodeVector{trunc_div}, ParameterVector{y});

        manager.register_pass<ov::intel_cpu::FuseFloorDivide>();
        manager.run_passes(model);
    }

    {
        auto x_const = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto y = make_shared<v0::Parameter>(element::i32, Shape{1});
        auto div = make_shared<v1::Divide>(x_const, y, false);
        model_ref = make_shared<Model>(NodeVector{div}, ParameterVector{y});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    FunctionsComparator::Result result = func_comparator(model_ref, model);
    ASSERT_FALSE(result.valid) << result.message;
}
