// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/nop_elimination.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace std;

TEST(nop_elimination, eliminate_convert) {
    std::shared_ptr<ov::Model> f;
    {
        Shape shape{};
        auto type = element::f32;
        auto A = make_shared<op::v0::Parameter>(type, shape);
        auto c = make_shared<op::v0::Convert>(A, element::f32);
        f = make_shared<ov::Model>(make_shared<op::v0::Abs>(c), ParameterVector{A});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(nop_elimination, convert_type_agnostic) {
    Shape shape{};
    std::shared_ptr<ov::Model> f;
    {
        auto A = make_shared<op::v0::Parameter>(ov::element::i8, shape);
        auto c1 = make_shared<op::v0::Convert>(A, ov::element::u8);
        auto c = make_shared<op::v0::Convert>(c1, ov::element::f32);
        auto z = make_shared<op::v3::NonZero>(c);
        f = make_shared<ov::Model>(make_shared<op::v0::Abs>(z), ParameterVector{A});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

template <typename Op>
void test_nop_eliminate_broadcast() {
    std::shared_ptr<ov::Model> f;
    {
        Shape shape{1};
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto b = make_shared<Op>(A, op::v0::Constant::create(element::u64, Shape{1}, {1}));
        f = make_shared<ov::Model>(make_shared<op::v0::Abs>(b), ParameterVector{A});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<Op>(f), 0);
}

TEST(nop_elimination, eliminate_broadcast_v1) {
    test_nop_eliminate_broadcast<op::v1::Broadcast>();
}

TEST(nop_elimination, eliminate_broadcast_v3) {
    test_nop_eliminate_broadcast<op::v3::Broadcast>();
}

TEST(nop_elimination, pass_property) {
    auto pass = std::make_shared<ov::pass::NopElimination>();
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(nop_elimination, reshape_elimination_v1) {
    auto generate_func = [](bool zero) {
        auto arg = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{8, 16, 2, 3});
        auto pattern_org = op::v0::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto pattern = op::v0::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto reshape_v1_org = std::make_shared<op::v1::Reshape>(arg, pattern_org, zero);
        auto reshape_v1 = std::make_shared<op::v1::Reshape>(reshape_v1_org, pattern, zero);
        auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
        return std::make_shared<ov::Model>(NodeVector{abs}, ParameterVector{arg});
    };

    auto func = generate_func(false);
    auto nopass_func = generate_func(false);
    auto func_zero = generate_func(true);
    auto nopass_func_zero = generate_func(true);

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(func);
    pass_manager.run_passes(func_zero);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func_zero) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func_zero) == 1);
}

TEST(nop_elimination, reshape_v1_1D) {
    auto make_model = [](int64_t input_dim, int64_t requested_dim) {
        const auto input = make_shared<op::v0::Parameter>(element::i64, PartialShape{{input_dim}});
        const auto abs = make_shared<op::v0::Abs>(input);
        const auto req_shape = op::v0::Constant::create(element::i64, Shape{1}, {requested_dim});
        const auto reshape = make_shared<op::v1::Reshape>(abs, req_shape, false);
        return make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    };
    // clang-format off
    vector<shared_ptr<ov::Model>> models{
        make_model( 7,  7),
        make_model( 7, -1),
        make_model(-1, -1),
    };
    // clang-format on

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    for (auto&& m : models) {
        pass_manager.run_passes(m);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(m), 0);
    }
}

TEST(nop_elimination, squeeze_reshape_elimination_check_info) {
    std::shared_ptr<ov::Model> f;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{8, 16, 1, 3});

        auto relu = std::make_shared<ov::op::v0::Relu>(arg);
        relu->set_friendly_name("relu");

        auto squeeze_axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(relu, squeeze_axes);
        squeeze->set_friendly_name("squeeze");

        auto reshape_shape = ov::op::v0::Constant::create(element::i64, Shape{4}, {8, 16, 1, 3});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(squeeze, reshape_shape, false);
        reshape->set_friendly_name("reshape");

        auto abs = std::make_shared<ov::op::v0::Abs>(reshape);

        f = std::make_shared<ov::Model>(NodeVector{abs}, ParameterVector{arg});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    bool movement_are_missing = true;
    for (auto node : f->get_ops()) {
        if (node->get_friendly_name() == "reshape" || node->get_friendly_name() == "squeeze") {
            movement_are_missing = false;
        }
    }
    ASSERT_TRUE(movement_are_missing);
}

TEST(nop_elimination, squeeze_unsqueeze_elimination) {
    std::shared_ptr<ov::Model> f;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{8, 16, 1, 3});

        auto relu = std::make_shared<ov::op::v0::Relu>(arg);
        relu->set_friendly_name("relu");

        auto squeeze_axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(relu, squeeze_axes);
        squeeze->set_friendly_name("squeeze");

        auto unsqueeze_axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(squeeze, unsqueeze_axes);
        unsqueeze->set_friendly_name("unsqueeze");

        auto abs = std::make_shared<ov::op::v0::Abs>(unsqueeze);

        f = std::make_shared<ov::Model>(NodeVector{abs}, ParameterVector{arg});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    bool movement_are_missing = true;
    for (auto node : f->get_ops()) {
        if (node->get_friendly_name() == "squeeze" || node->get_friendly_name() == "unsqueeze") {
            movement_are_missing = false;
        }
    }
    ASSERT_TRUE(movement_are_missing);
}

TEST(nop_elimination, squeeze_unsqueeze_elimination_dynamic_without_squeeze_axis) {
    std::shared_ptr<ov::Model> f;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 16, 1, 3});

        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(arg);
        squeeze->set_friendly_name("squeeze");

        auto unsqueeze_axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(squeeze, unsqueeze_axes);
        unsqueeze->set_friendly_name("unsqueeze");

        f = std::make_shared<ov::Model>(NodeVector{unsqueeze}, ParameterVector{arg});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    EXPECT_NO_THROW(pass_manager.run_passes(f));
}

TEST(nop_elimination, reshape_elimination_v1_dynamic_negative) {
    auto arg = std::make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::v0::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);
    auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
    auto f = std::make_shared<ov::Model>(NodeVector{abs}, ParameterVector{arg, pattern});
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 1);
}

TEST(nop_elimination, reshape_arithmetical_reduce_elimination_dynamic) {
    auto arg = std::make_shared<op::v0::Parameter>(element::i64, PartialShape({-1, 96, 100, 100}));
    auto reduce_axes = ov::op::v0::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce = std::make_shared<op::v1::ReduceMean>(arg, reduce_axes, true);
    auto pattern = op::v0::Constant::create(element::i64, Shape{4}, {0, 96, 1, 1});
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(reduce, pattern, true);
    auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
    auto f = std::make_shared<ov::Model>(NodeVector{abs}, ParameterVector{arg});
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>(false);
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 0);
}

TEST(nop_elimination, reshape_logical_reduce_elimination_dynamic) {
    auto arg = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape({-1, 96, 100, 100}));
    auto reduce_axes = ov::op::v0::Constant::create(element::i64, Shape{2}, {2, 3});
    auto reduce = std::make_shared<op::v1::ReduceLogicalAnd>(arg, reduce_axes, true);
    auto pattern = op::v0::Constant::create(element::i64, Shape{4}, {0, 96, 1, 1});
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(reduce, pattern, true);
    auto nz = std::make_shared<op::v3::NonZero>(reshape_v1);
    auto f = std::make_shared<ov::Model>(NodeVector{nz}, ParameterVector{arg});
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>(false);
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 0);
}

TEST(nop_elimination, reshape_elimination_v1_check_consumer_count) {
    std::shared_ptr<ov::Model> f;
    {
        auto arg = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{8, 16, 1, 3});

        auto reshape_1_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {128, 3});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(arg, reshape_1_shape, false);
        reshape_1->set_friendly_name("reshape_1");

        auto reshape_2_shape = ov::op::v0::Constant::create(element::i64, Shape{4}, {8, 16, 1, 3});
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(reshape_1, reshape_2_shape, false);
        reshape_2->set_friendly_name("reshape_2");

        auto relu = std::make_shared<ov::op::v0::Relu>(reshape_1);
        relu->set_friendly_name("relu");

        f = std::make_shared<ov::Model>(NodeVector{reshape_2, relu}, ParameterVector{arg});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 2);
}

TEST(nop_elimination, concat_elimination_single_node) {
    int64_t a = 0;
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto f = make_shared<ov::Model>(make_shared<op::v0::Concat>(NodeVector{A}, a), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 1);
}

TEST(nop_elimination, concat_elimination_single_input) {
    int64_t a = 0;
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, concat_elimination_single_input_dynamic) {
    int64_t a = 0;
    auto A = make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, unsqueeze_elimination) {
    const auto axis = op::v0::Constant::create<int64_t>(element::i64, {}, {0});
    const auto A =
        make_shared<op::v0::Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), Dimension::dynamic()});
    const auto unsqueeze = make_shared<op::v0::Unsqueeze>(A, axis);
    auto f = make_shared<ov::Model>(unsqueeze, ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(f), 1);
}

TEST(nop_elimination, squeeze_unsqueeze_overlap_elimination) {
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& sq_axes_val,
                            const std::vector<int64_t>& unsq_axes_val,
                            bool sq_to_unsq,
                            bool i32,
                            bool multiout,
                            size_t sc,
                            size_t usc,
                            size_t rc) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> sq_axes;
        shared_ptr<Node> unsq_axes;
        if (i32) {
            std::vector<int32_t> sq_axes_val_i32(sq_axes_val.size());
            std::vector<int32_t> unsq_axes_val_i32(unsq_axes_val.size());
            std::transform(sq_axes_val.begin(), sq_axes_val.end(), sq_axes_val_i32.begin(), [](int64_t x) {
                return (int32_t)x;
            });
            std::transform(unsq_axes_val.begin(), unsq_axes_val.end(), unsq_axes_val_i32.begin(), [](int64_t x) {
                return (int32_t)x;
            });

            sq_axes = op::v0::Constant::create<int32_t>(element::i32, Shape{sq_axes_val.size()}, sq_axes_val_i32);
            unsq_axes = op::v0::Constant::create<int32_t>(element::i32, Shape{unsq_axes_val.size()}, unsq_axes_val_i32);
        } else {
            sq_axes = op::v0::Constant::create<int64_t>(element::i64, Shape{sq_axes_val.size()}, sq_axes_val);
            unsq_axes = op::v0::Constant::create<int64_t>(element::i64, Shape{unsq_axes_val.size()}, unsq_axes_val);
        }

        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        shared_ptr<Node> A1;
        if (multiout) {
            shared_ptr<Node> k;
            auto last_dim = shape.rank().get_length() - 1;
            if (shape[last_dim].is_dynamic()) {
                k = make_shared<op::v1::Gather>(make_shared<op::v0::ShapeOf>(A),
                                                op::v0::Constant::create(element::i64, {}, {last_dim}),
                                                op::v0::Constant::create(element::i64, {}, {0}));
            } else {
                k = make_shared<op::v0::Constant>(element::i64,
                                                  Shape{},
                                                  std::vector<int64_t>{shape[last_dim].get_length()});
            }
            A1 = make_shared<op::v1::TopK>(A, k, last_dim, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::NONE);
        } else {
            A1 = make_shared<op::v0::Abs>(A);
        }

        shared_ptr<Node> B1;
        if (sq_to_unsq) {
            auto B = make_shared<op::v0::Squeeze>((multiout ? A1->output(0) : A1), sq_axes);
            B1 = make_shared<op::v0::Unsqueeze>(B, unsq_axes);
        } else {
            auto B = make_shared<op::v0::Unsqueeze>((multiout ? A1->output(0) : A1), unsq_axes);
            B1 = make_shared<op::v0::Squeeze>(B, sq_axes);
        }

        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();

        pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::Validate>();
        pass_manager.register_pass<ov::pass::NopElimination>();
        pass_manager.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), sc) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), usc) << casename;
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), rc) << casename;
    };

    // static shapes, all squeeze/unsqueeze replaced by reshape
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, false, false, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, true, false, 0, 0, 1);
    // multioutout ops
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, false, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{1}, {0}, {0, 1, 2, 3}, true, true, true, 0, 0, 1);

    // axes match - expect all squeeze/unsqueeze/reshape cancel out
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {1, 2}, true, true, true, 0, 0, 0);

    // dynamic shapes - axes match, expect all cancel
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1},
                  {0, 2, 4},
                  {0, 2, 4},
                  true,
                  true,
                  true,
                  0,
                  0,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 2, 1}, {0, 2, 4}, {0, 2, 4}, true, false, true, 0, 0, 0);

    // squeeze axes overlap fully
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 3}, {1, 2}, {1, 2, 3}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {1, 2},
                  {1, 2, 3},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);
    check_usecase(PartialShape{2, 1, 1, 4}, {1, 2}, {1, 2, 3}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, Dimension::dynamic(), Dimension::dynamic()},
                  {1, 2},
                  {1, 2, 3},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {2, 3},
                  {2, 3, 5},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);

    // unsqueeze axes overlap fully
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 1, 1, Dimension::dynamic(), 3},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 3, 1, 1}, {2, 3}, {2}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{3, 1, 1}, {1, 2}, {1}, true, true, true, 0, 0, 1);

    // squeeze->unsqueeze axes overlap
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 4}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {1, 2},
                  {0},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{3, 1, 1, 4}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, Dimension::dynamic(), Dimension::dynamic()},
                  {1, 2},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 3, Dimension::dynamic(), 4},
                  {1, 2},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{2, 1, Dimension::dynamic(), 1, Dimension::dynamic()},
                  {1, 3},
                  {3},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 1, 4},
                  {4, 5},
                  {1, 5},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);

    //
    // Unsqueeze->Squeeze cases, testcase 23 - ..
    //
    // static shapes, all unsqueeze/squeeze replaced by reshape
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, false, false, 0, 0, 1);
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, true, false, 0, 0, 1);
    // multioutout ops
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, false, true, 0, 0, 1);
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, true, true, 0, 0, 1);
    check_usecase(PartialShape{1}, {0}, {0, 1, 2, 3}, false, true, true, 0, 0, 1);
    check_usecase(PartialShape{3, 1, 1, 4}, {2, 3}, {0}, false, true, true, 0, 0, 1);
    // dynamic shapes
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1},
                  {0, 2, 4},
                  {0, 2, 4},
                  false,
                  true,
                  true,
                  0,
                  0,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()}, {2}, {0}, true, true, true, 1, 1, 0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 4}, {2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
}

TEST(nop_elimination, squeeze_squeeze_overlap_elimination) {
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& sq_axes_val_1,
                            const std::vector<int64_t>& sq_axes_val_2,
                            size_t sq) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);
        auto sq_axes_1 = op::v0::Constant::create<int64_t>(element::i64, Shape{sq_axes_val_1.size()}, sq_axes_val_1);
        auto sq_axes_2 = op::v0::Constant::create<int64_t>(element::i64, Shape{sq_axes_val_2.size()}, sq_axes_val_2);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Squeeze>(A1, sq_axes_1);
        auto B1 = make_shared<op::v0::Squeeze>(B, sq_axes_2);
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();

        pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::Validate>();
        pass_manager.register_pass<ov::pass::NopElimination>();
        pass_manager.run_passes(optimized_f);
        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 2) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), sq) << casename;
    };

    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic()}, {0}, {1}, 1);
    check_usecase(PartialShape{1, 1, 1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {2, 1}, {2, 4}, 1);
    check_usecase(PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 1, 1}, {-1, -5}, {2}, 1);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {0}, {1, 3}, 1);
}

TEST(nop_elimination, unsqueeze_unsqueeze_overlap_elimination) {
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& unsq_axes_val_1,
                            const std::vector<int64_t>& unsq_axes_val_2,
                            size_t unsq) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);
        auto unsq_axes_1 =
            op::v0::Constant::create<int64_t>(element::i64, Shape{unsq_axes_val_1.size()}, unsq_axes_val_1);
        auto unsq_axes_2 =
            op::v0::Constant::create<int64_t>(element::i64, Shape{unsq_axes_val_2.size()}, unsq_axes_val_2);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Unsqueeze>(A1, unsq_axes_1);
        auto B1 = make_shared<op::v0::Unsqueeze>(B, unsq_axes_2);
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();

        pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::Validate>();
        pass_manager.register_pass<ov::pass::NopElimination>();
        pass_manager.run_passes(optimized_f);
        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 2) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), unsq) << casename;
    };

    check_usecase(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()}, {0}, {2}, 1);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {2, 1}, {2, 4}, 1);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1}, {-1, -3}, {2}, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {0}, {1, 3}, 1);
}

TEST(nop_elimination, unsqueeze_squeeze_elimination) {
    auto generate_func = [](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Unsqueeze>(A1, axes);
        auto B1 = make_shared<op::v0::Squeeze>(B, axes);
        return make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
    };

    auto check_usecase = [&](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto baseline_f = generate_func(shape, axes_val);
        auto optimized_f = generate_func(shape, axes_val);
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{6}, std::vector<int64_t>{0});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 3});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 2, 4});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{-1, -4});
}

TEST(nop_elimination, reshape_unsqueeze_elimination) {
    auto check_usecase =
        [](const Shape& shape, const std::vector<int64_t>& pat_val, bool zero, const std::vector<int64_t>& axes_val) {
            auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
            auto pat = op::v0::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
            auto A = make_shared<op::v0::Parameter>(element::f32, shape);
            auto A1 = make_shared<op::v0::Abs>(A);

            auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
            auto pat2 = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
            auto B1 = make_shared<op::v0::Unsqueeze>(B, axes);
            auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
            auto optimized_f = baseline_f->clone();
            pass::Manager manager;
            manager.register_pass<ov::pass::NopElimination>();
            manager.run_passes(optimized_f);

            ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
            ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
            ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
        };

    check_usecase(Shape{1, 2, 3, 2, 1}, {2, 3, 2}, false, {2, 4});
    check_usecase(Shape{12}, {2, 3, 2}, false, {3});
    check_usecase(Shape{3, 2, 1, 2}, {0, 2, 2}, true, {1, 4});
    check_usecase(Shape{2, 3, 2}, {2, -1, 2}, false, {2});
    check_usecase(Shape{2, 3, 2, 1}, {2, 3, 2}, false, {0});
}
TEST(nop_elimination, reshape_squeeze_elimination) {
    auto check_usecase =
        [](const Shape& shape, const std::vector<int64_t>& pat_val, bool zero, const std::vector<int64_t>& axes_val) {
            auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
            auto pat = op::v0::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
            auto A = make_shared<op::v0::Parameter>(element::f32, shape);
            auto A1 = make_shared<op::v0::Abs>(A);

            auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
            auto pat2 = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
            auto B1 = make_shared<op::v0::Squeeze>(B, axes);
            auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
            auto optimized_f = baseline_f->clone();
            pass::Manager manager;
            manager.register_pass<ov::pass::NopElimination>();
            manager.run_passes(optimized_f);

            ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
            ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
            ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
        };

    check_usecase(Shape{1, 2, 3, 2, 1}, {2, 3, 1, 2, 1}, false, {2, 4});
    check_usecase(Shape{12}, {2, 3, 2, 1}, false, {3});
    check_usecase(Shape{3, 2, 1, 2}, {0, 1, 2, 2, 1}, true, {1, 4});
    check_usecase(Shape{2, 3, 2}, {2, -1, 1, 2}, false, {2});
    check_usecase(Shape{2, 3, 2, 1}, {1, 2, 3, 2}, false, {0});
}

TEST(nop_elimination, reshape_reshape_elimination) {
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& pat_val, bool zero) {
        auto pat = op::v0::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
        auto pat2 = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, true);
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 2);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, std::vector<int64_t>{2, 3, 2}, false);
    check_usecase(Shape{12}, std::vector<int64_t>{2, 3, 2}, false);
    check_usecase(Shape{3, 2, 1, 2}, std::vector<int64_t>{0, 2, 2}, true);
    check_usecase(Shape{2, 3, 2}, ::vector<int64_t>{2, -1, 2}, false);
    check_usecase(Shape{2, 3, 2, 1}, ::vector<int64_t>{2, 3, 2}, false);
}

TEST(nop_elimination, squeeze_reshape_elimination) {
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& indices_val) {
        auto indices = op::v0::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v0::Squeeze>(A1, indices);
        auto pat2 = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, false);
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, std::vector<int64_t>{0, 4});
    check_usecase(Shape{1, 1}, std::vector<int64_t>{0, 1});
    check_usecase(Shape{2, 3, 1, 2}, std::vector<int64_t>{2});
    check_usecase(Shape{1, 6, 2, 1}, std::vector<int64_t>{3});
}

TEST(nop_elimination, unsqueeze_reshape_elimination) {
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& indices_val) {
        auto indices = op::v0::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
        auto A = make_shared<op::v0::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v0::Unsqueeze>(A1, indices);
        auto pat2 = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, false);
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = baseline_f->clone();
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{2, 3, 2}, std::vector<int64_t>{0, 4});
    check_usecase(Shape{}, std::vector<int64_t>{0, 1});
    check_usecase(Shape{2, 3, 2}, std::vector<int64_t>{2});
    check_usecase(Shape{1, 6, 2}, std::vector<int64_t>{3});
}

TEST(nop_elimination, squeeze_unsqueeze_elimination_negative) {
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& indices_val) {
        auto indices = op::v0::Constant::create(element::i64, Shape{indices_val.size()}, indices_val);
        auto input = make_shared<op::v0::Parameter>(element::f32, shape);
        auto squeeze = make_shared<ov::op::v0::Squeeze>(input, indices);
        auto baseline_f = make_shared<ov::Model>(squeeze, ParameterVector{input});
        auto optimized_f = baseline_f->clone();
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);
        ASSERT_EQ(count_ops_of_type<ov::op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<ov::op::v0::Squeeze>(optimized_f), 1);
    };

    check_usecase(Shape{1, 1, 1}, std::vector<int64_t>{0, 1, 2});
}

TEST(nop_elimination, topk_convert_elimination) {
    auto check_usecase = []() {
        auto A = make_shared<op::v0::Parameter>(element::f32, Shape{20, 3, 4});
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v1::TopK>(A1,
                                           op::v0::Constant::create(element::i64, {}, {10}),
                                           0,
                                           op::v1::TopK::Mode::MAX,
                                           op::v1::TopK::SortType::NONE);
        auto C = make_shared<op::v0::Convert>(B->output(0), B->output(0).get_element_type());
        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(C), ParameterVector{A});
        auto optimized_f = baseline_f->clone();
        pass::Manager manager;
        manager.register_pass<ov::pass::NopElimination>();
        manager.run_passes(optimized_f);

        ASSERT_EQ(count_ops_of_type<op::v0::Convert>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Convert>(optimized_f), 0);
    };

    check_usecase();
}

TEST(nop_elimination, gather_3d_indices_constant_axis_1) {
    auto check_usecase = [](const PartialShape& pshape,
                            bool i32,
                            bool multiout,
                            const std::vector<int64_t>& indices_val,
                            int64_t axis_val,
                            size_t num) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> indices;
        shared_ptr<Node> axis;
        if (i32) {
            std::vector<int32_t> indices_val_i32(indices_val.size());
            std::transform(indices_val.begin(), indices_val.end(), indices_val_i32.begin(), [](int64_t x) {
                return (int32_t)x;
            });
            indices = op::v0::Constant::create<int32_t>(element::i32, Shape{indices_val.size()}, indices_val_i32);
            axis = op::v0::Constant::create<int32_t>(element::i32, Shape{}, {(int32_t)axis_val});
        } else {
            indices = op::v0::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
            axis = op::v0::Constant::create<int64_t>(element::i64, Shape{}, {axis_val});
        }

        auto A = make_shared<op::v0::Parameter>(element::f32, pshape);
        shared_ptr<Node> A1;
        if (multiout) {
            auto last_dim = pshape.rank().get_length() - 1;
            A1 = make_shared<op::v1::TopK>(A,
                                           op::v0::Constant::create(element::i64, {}, {1}),
                                           last_dim,
                                           op::v1::TopK::Mode::MAX,
                                           op::v1::TopK::SortType::NONE);
        } else {
            A1 = make_shared<op::v0::Abs>(A);
        }
        auto G = make_shared<op::v1::Gather>((multiout ? A1->output(0) : A1), indices, axis);

        auto baseline_f = make_shared<ov::Model>(make_shared<op::v0::Abs>(G), ParameterVector{A});
        auto optimized_f = baseline_f->clone();

        pass::Manager pass_manager;
        pass_manager.register_pass<ov::pass::Validate>();
        pass_manager.register_pass<ov::pass::NopElimination>();
        pass_manager.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v1::Gather>(baseline_f), 1) << casename;
        // the pass should short cut the Gather i/p with the gather users
        // since we are fetching the whole tensor using gather op
        ASSERT_EQ(count_ops_of_type<op::v1::Gather>(optimized_f), num) << casename;
    };
    for (auto& i32 : {true, false})
        for (auto& multiout : {true, false}) {
            check_usecase(PartialShape{1, 3, 2}, i32, multiout, std::vector<int64_t>{1}, 0, 0);
            check_usecase(PartialShape{3, 2, 1}, i32, multiout, std::vector<int64_t>{0, 1}, 1, 0);
            check_usecase(PartialShape{3, 2, 1}, i32, multiout, std::vector<int64_t>{1}, 2, 0);
            check_usecase(PartialShape{1, 16}, i32, multiout, std::vector<int64_t>{0, 0}, 0, 1);
        }
}

struct ShapeParams {
    PartialShape shape1;
    Shape shape2;
    bool swap_inputs;
    bool can_fuse;
};

enum class OpType {
    ADD,
    SUBTRACT,
    SUBTRACT_WITH_CONVERT,
    MULTIPLY,
    DIVIDE,
};

static std::ostream& operator<<(std::ostream& os, OpType kind) {
    switch (kind) {
    case OpType::ADD:
        os << "add";
        break;
    case OpType::SUBTRACT:
        os << "subtract";
        break;
    case OpType::SUBTRACT_WITH_CONVERT:
        os << "subtract_with_convert";
        break;
    case OpType::MULTIPLY:
        os << "multiply";
        break;
    case OpType::DIVIDE:
        os << "divide";
        break;
    }
    return os;
}

enum class ConstantKind {
    ZERO,
    ONE,
    RANDOM,
};

static std::ostream& operator<<(std::ostream& os, ConstantKind kind) {
    switch (kind) {
    case ConstantKind::ZERO:
        os << "zero";
        break;
    case ConstantKind::ONE:
        os << "one";
        break;
    case ConstantKind::RANDOM:
        os << "random";
        break;
    }
    return os;
}

struct TypeParams {
    OpType op_type;
    ConstantKind constant_kind;
    bool can_fuse;
};

using EliminateEltwiseParams = std::tuple<ShapeParams, TypeParams, element::Type>;

class EliminateEltwiseTests : public testing::WithParamInterface<EliminateEltwiseParams>,
                              virtual public TransformationTestsF {
public:
    static std::string get_test_case_name(testing::TestParamInfo<EliminateEltwiseParams> info) {
        const auto& shape_params = std::get<0>(info.param);
        const auto& type_params = std::get<1>(info.param);
        const auto& element_type = std::get<2>(info.param);
        std::ostringstream result;
        result << type_params.op_type << "_input1=" << shape_params.shape1 << "_input2=" << shape_params.shape2
               << "_swap_inputs=" << std::boolalpha << shape_params.swap_inputs
               << "_constant=" << type_params.constant_kind << "_type=" << element_type;
        return result.str();
    }
};

std::vector<element::Type> types{
    element::f32,
    element::f16,
    element::f64,
    element::i32,
    element::u32,
    element::i64,
    element::u64,
    element::i8,
    element::u8,
    element::i16,
    element::u16,
};

TEST_P(EliminateEltwiseTests, eliminate_eltwise) {
    auto params = GetParam();
    const auto& shape_params = std::get<0>(params);
    const auto& type_params = std::get<1>(params);
    const auto& type = std::get<2>(params);
    const auto& shape1 = shape_params.shape1;
    const auto& shape2 = shape_params.shape2;
    bool swap_inputs = shape_params.swap_inputs;
    bool can_fuse = shape_params.can_fuse && type_params.can_fuse;

    auto parameter = make_shared<op::v0::Parameter>(type, shape1);

    auto constant_type = type;
    if (type_params.op_type == OpType::SUBTRACT_WITH_CONVERT) {
        if (type == types[0])
            constant_type = types[1];
        else
            constant_type = types[0];
    }

    std::shared_ptr<Node> constant;
    switch (type_params.constant_kind) {
    case ConstantKind::ZERO:
        constant = op::v0::Constant::create(constant_type, shape2, {0});
        break;
    case ConstantKind::ONE:
        constant = op::v0::Constant::create(constant_type, shape2, {1});
        break;
    case ConstantKind::RANDOM:
        int32_t resolution = 1;
        if (constant_type.is_real()) {
            resolution = 1000;
        }
        constant = ov::test::utils::make_constant(constant_type,
                                                  shape2,
                                                  ov::test::utils::InputGenerateData(2, 18, resolution));
        break;
    }

    if (type_params.op_type == OpType::SUBTRACT_WITH_CONVERT) {
        constant = std::make_shared<ov::op::v0::Convert>(constant, type);
    }

    shared_ptr<Node> A = parameter;
    shared_ptr<Node> B = constant;
    if (swap_inputs) {
        std::swap(A, B);
        if (type_params.op_type == OpType::SUBTRACT || type_params.op_type == OpType::SUBTRACT_WITH_CONVERT ||
            type_params.op_type == OpType::DIVIDE) {
            can_fuse = false;
        }
    }

    shared_ptr<Node> node;
    switch (type_params.op_type) {
    case OpType::ADD:
        node = make_shared<ov::op::v1::Add>(A, B);
        break;
    case OpType::SUBTRACT:
    case OpType::SUBTRACT_WITH_CONVERT:
        node = make_shared<ov::op::v1::Subtract>(A, B);
        break;
    case OpType::MULTIPLY:
        node = make_shared<ov::op::v1::Multiply>(A, B);
        break;
    case OpType::DIVIDE:
        node = make_shared<ov::op::v1::Divide>(A, B);
        break;
    default:
        ASSERT_FALSE(true) << "Invalid OpType";
    }
    auto abs = make_shared<ov::op::v0::Abs>(node);
    model = make_shared<ov::Model>(abs, ParameterVector{parameter});

    manager.register_pass<ov::pass::NopElimination>();

    if (can_fuse) {
        auto abs = make_shared<ov::op::v0::Abs>(parameter);
        model_ref = make_shared<ov::Model>(abs, ParameterVector{parameter});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    if (type == element::f32) {
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    }
}

std::vector<ShapeParams> shape_params = {
    // input 1, input 2, swap inputs, can fuse
    {Shape{}, Shape{}, false, true},
    {Shape{}, Shape{}, true, true},
    {Shape{5}, Shape{}, false, true},
    {Shape{5}, Shape{1}, false, true},
    {Shape{5}, Shape{5}, false, true},
    {Shape{5}, Shape{5}, true, true},
    {Shape{2, 3, 5}, Shape{}, false, true},
    {Shape{2, 3, 5}, Shape{1}, false, true},
    {Shape{2, 3, 5}, Shape{1, 1}, false, true},
    {Shape{2, 3, 5}, Shape{1, 1, 1}, false, true},
    {Shape{2, 3, 5}, Shape{5}, false, true},
    {Shape{2, 3, 5}, Shape{1, 5}, false, true},
    {Shape{2, 3, 5}, Shape{1, 1, 5}, false, true},
    {Shape{2, 3, 5}, Shape{3, 5}, false, true},
    {Shape{2, 3, 5}, Shape{1, 3, 5}, false, true},
    {Shape{2, 3, 5}, Shape{2, 3, 5}, false, true},
    {Shape{2, 3, 5}, Shape{2, 3, 5}, true, true},
    {PartialShape::dynamic(), Shape{}, false, true},
    {PartialShape::dynamic(3), Shape{}, false, true},
    {PartialShape::dynamic(3), Shape{1}, false, true},
    {PartialShape::dynamic(3), Shape{1, 1}, false, true},
    {PartialShape::dynamic(3), Shape{1, 1, 1}, false, true},
    {PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}, Shape{1, 1}, false, true},
    {PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}, Shape{3, 1}, false, true},
    {PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()}, Shape{1, 3, 1}, false, true},
    // negative cases
    {Shape{}, Shape{1}, false, false},
    {Shape{}, Shape{2, 3}, false, false},
    {Shape{5}, Shape{1, 1}, false, false},
    {Shape{4, 1, 3}, Shape{2, 3}, false, false},
    {Shape{1, 2, 3}, Shape{4, 2, 3}, false, false},
    {Shape{1, 1, 3}, Shape{2, 1, 3}, false, false},
    {Shape{1, 2, 3}, Shape{1, 1, 1, 1}, false, false},
    {PartialShape::dynamic(), Shape{2, 3, 4}, false, false},
    {PartialShape::dynamic(3), Shape{1, 2, 1}, false, false},
    {PartialShape::dynamic(3), Shape{1, 1, 1, 1}, false, false},
};

std::vector<TypeParams> type_params = {
    // op type, constant value, can fuse
    {OpType::ADD, ConstantKind::ZERO, true},
    {OpType::ADD, ConstantKind::RANDOM, false},
    {OpType::SUBTRACT, ConstantKind::ZERO, true},
    {OpType::SUBTRACT, ConstantKind::RANDOM, false},
    {OpType::SUBTRACT_WITH_CONVERT, ConstantKind::ZERO, true},
    {OpType::SUBTRACT_WITH_CONVERT, ConstantKind::RANDOM, false},
    {OpType::MULTIPLY, ConstantKind::ONE, true},
    {OpType::MULTIPLY, ConstantKind::RANDOM, false},
    {OpType::DIVIDE, ConstantKind::ONE, true},
    {OpType::DIVIDE, ConstantKind::RANDOM, false},
};

INSTANTIATE_TEST_SUITE_P(EliminateEltwise,
                         EliminateEltwiseTests,
                         ::testing::Combine(::testing::ValuesIn(shape_params),
                                            ::testing::ValuesIn(type_params),
                                            ::testing::ValuesIn(types)),
                         EliminateEltwiseTests::get_test_case_name);

TEST_F(TransformationTestsF, eliminate_eltwise_dequantization_subgraph) {
    {
        auto constant = ov::op::v0::Constant::create(element::i8, Shape{}, {2});
        auto convert = make_shared<ov::op::v0::Convert>(constant, element::f32);
        auto sub = make_shared<ov::op::v1::Subtract>(convert, ov::op::v0::Constant::create(element::f32, Shape{}, {0}));
        auto mul = make_shared<ov::op::v1::Multiply>(sub, ov::op::v0::Constant::create(element::f32, Shape{}, {1}));
        model = make_shared<ov::Model>(mul, ParameterVector{});
    }
    {
        auto constant = ov::op::v0::Constant::create(element::i8, Shape{}, {2});
        auto convert = make_shared<ov::op::v0::Convert>(constant, element::f32);
        auto mul = make_shared<ov::op::v1::Multiply>(convert, ov::op::v0::Constant::create(element::f32, Shape{}, {1}));
        model_ref = make_shared<ov::Model>(mul, ParameterVector{});
    }

    manager.register_pass<ov::pass::NopElimination>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

enum class SplitType { Split, VariadicSplit };

enum class RNNType : size_t {
    NONE = 0,
    RNN = 1,
    GRU = 3,
    LSTM = 4,
};

struct SplitConcatEliminationParams {
    SplitType split_type;
    RNNType rnn_type;
    size_t seq_len;  // must be divisible by split_len
    size_t split_len;
    int64_t split_axis;
    int64_t concat_axis;
};

class SplitConcatElimination : public testing::WithParamInterface<SplitConcatEliminationParams>,
                               public ov::test::TestsCommon {};

TEST_P(SplitConcatElimination, eliminate_split_concat_subgraph) {
    const auto& p = GetParam();
    size_t batch = 2;
    size_t input_size = 4;
    size_t hidden_size = 2;
    size_t seq_len = p.seq_len;
    size_t num_dir = 1;

    EXPECT_TRUE(seq_len % p.split_len == 0) << "Seq_len must be divisible by split_len.";

    ParameterVector params;
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch, seq_len, input_size});

    shared_ptr<Node> data = param;
    shared_ptr<Node> sequence;
    auto gate = static_cast<size_t>(p.rnn_type);
    auto axis_const = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, p.split_axis);
    auto H = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch, num_dir, hidden_size});
    auto C = make_shared<ov::op::v0::Parameter>(element::f32, Shape{batch, num_dir, hidden_size});
    auto seq_lengths = make_shared<ov::op::v0::Parameter>(element::i64, Shape{batch});
    auto W = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_dir, gate * hidden_size, input_size});
    auto R = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_dir, gate * hidden_size, hidden_size});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{num_dir, gate * hidden_size});
    auto direction = op::RecurrentSequenceDirection::FORWARD;
    if (p.rnn_type == RNNType::RNN) {
        sequence = make_shared<ov::op::v5::RNNSequence>(data, H, seq_lengths, W, R, B, hidden_size, direction);
        data = make_shared<ov::op::v0::Squeeze>(sequence->output(0), axis_const);
        params = {H, seq_lengths, W, R, B};
    } else if (p.rnn_type == RNNType::GRU) {
        sequence = make_shared<ov::op::v5::GRUSequence>(data, H, seq_lengths, W, R, B, hidden_size, direction);
        data = make_shared<ov::op::v0::Squeeze>(sequence->output(0), axis_const);
        params = {H, seq_lengths, W, R, B};
    } else if (p.rnn_type == RNNType::LSTM) {
        sequence = make_shared<ov::op::v5::LSTMSequence>(data, H, C, seq_lengths, W, R, B, hidden_size, direction);
        data = make_shared<ov::op::v0::Squeeze>(sequence->output(0), axis_const);
        params = {H, C, seq_lengths, W, R, B};
    }
    params.push_back(param);

    shared_ptr<ov::Node> split;
    if (p.split_type == SplitType::Split) {
        split = make_shared<ov::op::v1::Split>(data->output(0), axis_const, p.seq_len / p.split_len);
    } else if (p.split_type == SplitType::VariadicSplit) {
        auto split_lengths = make_shared<ov::op::v0::Constant>(element::i64,
                                                               Shape{seq_len / p.split_len},
                                                               std::vector<size_t>(seq_len / p.split_len, p.split_len));
        split = make_shared<ov::op::v1::VariadicSplit>(data->output(0), axis_const, split_lengths);
    }

    auto outputs_to_concat = split->outputs();
    if (sequence) {
        outputs_to_concat[outputs_to_concat.size() - 1] = sequence->output(1);
    }
    auto concat = make_shared<ov::op::v0::Concat>(outputs_to_concat, p.concat_axis);
    auto sigmoid = make_shared<ov::op::v0::Sigmoid>(concat);
    auto res = make_shared<ov::op::v0::Result>(sigmoid);
    auto model = make_shared<ov::Model>(ResultVector{res}, ParameterVector{params});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(model);

    // the transformation won't be applied if split_len is not equal to 1
    size_t expect_concat = p.split_len == 1 ? 0 : 1;
    size_t expect_split = p.split_len == 1 ? 0 : 1;
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), expect_concat)
        << "SplitConcatElimination transformation has failed. "
           "The number of Concat ops is not " +
               to_string(expect_concat);
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Split>(model) + count_ops_of_type<ov::op::v1::VariadicSplit>(model),
              expect_split)
        << "SplitConcatElimination transformation has failed. "
           "The number of Split/VariadicSplit ops is not " +
               to_string(expect_split);
}

static const vector<SplitConcatEliminationParams> params = {
    SplitConcatEliminationParams{SplitType::Split, RNNType::NONE, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::Split, RNNType::RNN, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::Split, RNNType::LSTM, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::Split, RNNType::GRU, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::Split, RNNType::GRU, 10, 2, 1, 1},
    SplitConcatEliminationParams{SplitType::Split, RNNType::NONE, 10, 1, -2, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::NONE, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::RNN, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::LSTM, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::GRU, 10, 1, 1, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::GRU, 10, 2, 1, 1},
    SplitConcatEliminationParams{SplitType::VariadicSplit, RNNType::NONE, 10, 1, 1, -2}};

INSTANTIATE_TEST_SUITE_P(SplitConcatElimination, SplitConcatElimination, testing::ValuesIn(params));

TEST(SplitConcatElimination, split_inputs_not_in_order) {
    int64_t axis = 1;
    auto axis_const = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, axis);

    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10});
    auto split = make_shared<ov::op::v1::Split>(param->output(0), axis_const, 10);
    OutputVector outputs_to_concat = split->outputs();

    // change order of inputs to Concat, in this case the transformation won't be applied
    std::reverse(outputs_to_concat.begin(), outputs_to_concat.end());
    auto concat = make_shared<ov::op::v0::Concat>(outputs_to_concat, axis);
    auto sigmoid = make_shared<ov::op::v0::Sigmoid>(concat);
    auto res = make_shared<ov::op::v0::Result>(sigmoid);
    auto model = make_shared<ov::Model>(ResultVector{res}, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(model);
    // the transformation shouldn't be applied
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 1) << "SplitConcatElimination transformation has failed. "
                                                                  "The number of Concat ops is not 1";
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Split>(model), 1) << "SplitConcatElimination transformation has failed. "
                                                                 "The number of Split ops is not 1";
}

TEST(SplitConcatElimination, no_sequence_found) {
    int64_t axis = 1;
    auto axis_const = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, axis);

    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10});
    auto param_2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});
    auto split = make_shared<ov::op::v1::Split>(param->output(0), axis_const, 10);
    OutputVector outputs_to_concat = split->outputs();

    outputs_to_concat[outputs_to_concat.size() - 1] = param_2;
    auto concat = make_shared<ov::op::v0::Concat>(outputs_to_concat, axis);
    auto sigmoid = make_shared<ov::op::v0::Sigmoid>(concat);
    auto res = make_shared<ov::op::v0::Result>(sigmoid);
    auto model = make_shared<ov::Model>(ResultVector{res}, ParameterVector{param, param_2});

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Validate>();
    pass_manager.register_pass<ov::pass::NopElimination>();
    pass_manager.run_passes(model);
    // the transformation shouldn't be applied
    EXPECT_EQ(count_ops_of_type<ov::op::v0::Concat>(model), 1) << "SplitConcatElimination transformation has failed. "
                                                                  "The number of Concat ops is not 1";
    EXPECT_EQ(count_ops_of_type<ov::op::v1::Split>(model), 1) << "SplitConcatElimination transformation has failed. "
                                                                 "The number of Split ops is not 1";
}

TEST(nop_elimination, gather_to_squeeze) {
    auto generate_func = [](int64_t gather_axis) {
        ov::Shape shape{3, 3, 4, 4};
        shape[gather_axis] = 1;
        auto arg = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto indices = op::v0::Constant::create(element::i64, Shape{}, vector<int64_t>{0});
        auto axis = op::v0::Constant::create(element::i64, Shape{}, vector<int64_t>{gather_axis});
        auto gather = std::make_shared<op::v8::Gather>(arg, indices, axis);
        return std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{arg});
    };

    auto func_axis_0 = generate_func(0);
    auto func_axis_1 = generate_func(1);
    auto func_axis_2 = generate_func(2);
    auto func_axis_3 = generate_func(3);
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    auto run_and_check = [&](std::shared_ptr<ov::Model>& func) {
        pass_manager.run_passes(func);
        EXPECT_EQ(count_ops_of_type<op::v8::Gather>(func), 0);
        EXPECT_EQ(count_ops_of_type<op::v0::Squeeze>(func), 1);
    };
    run_and_check(func_axis_0);
    run_and_check(func_axis_1);
    run_and_check(func_axis_2);
    run_and_check(func_axis_3);
}

TEST(nop_elimination, not_gather_to_squeeze_with_vector_indices) {
    auto generate_func = [](int64_t gather_axis) {
        ov::Shape shape{3, 3, 4, 4};
        shape[gather_axis] = 1;
        auto arg = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto indices = op::v0::Constant::create(element::i64, Shape{1, 1}, vector<int64_t>{0});
        auto axis = op::v0::Constant::create(element::i64, Shape{}, vector<int64_t>{gather_axis});
        auto gather = std::make_shared<op::v8::Gather>(arg, indices, axis);
        return std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{arg});
    };

    auto func_axis_0 = generate_func(0);
    auto func_axis_1 = generate_func(1);
    auto func_axis_2 = generate_func(2);
    auto func_axis_3 = generate_func(3);
    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::NopElimination>();
    auto run_and_check = [&](std::shared_ptr<ov::Model>& func) {
        pass_manager.run_passes(func);
        EXPECT_EQ(count_ops_of_type<op::v8::Gather>(func), 1);
        EXPECT_EQ(count_ops_of_type<op::v0::Squeeze>(func), 0);
    };
    run_and_check(func_axis_0);
    run_and_check(func_axis_1);
    run_and_check(func_axis_2);
    run_and_check(func_axis_3);
}

TEST_F(TransformationTestsF, Nopv1Broadcast) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto broadcast_shape = ov::op::v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1});
        auto broadcast = std::make_shared<op::v1::Broadcast>(data, broadcast_shape);
        auto relu = std::make_shared<op::v0::Relu>(broadcast);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateNopBroadcast>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, Nopv3Broadcast) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto broadcast_shape = ov::op::v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1});
        auto broadcast = std::make_shared<op::v3::Broadcast>(data, broadcast_shape);
        auto relu = std::make_shared<op::v0::Relu>(broadcast);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateNopBroadcast>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NopTile) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto repeats = ov::op::v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1});
        auto tile = std::make_shared<op::v0::Tile>(data, repeats);
        auto relu = std::make_shared<op::v0::Relu>(tile);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateNopBroadcast>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EliminateSliceBeforeGatherElements) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto start = ov::op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto stop = ov::op::v0::Constant::create(element::i32, Shape{1}, {2});
        auto step = ov::op::v0::Constant::create(element::i32, Shape{1}, {1});
        auto axis = ov::op::v0::Constant::create(element::i32, Shape{1}, {-1});
        auto slice = std::make_shared<op::v8::Slice>(data, start, stop, step, axis);

        auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{-1, -1, -1, -1});
        auto gather_elements = std::make_shared<op::v6::GatherElements>(slice, indices, 2);

        auto relu = std::make_shared<op::v0::Relu>(gather_elements);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data, indices});
        manager.register_pass<ov::pass::EliminateSliceBeforeGatherElements>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
        auto indices = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{-1, -1, -1, -1});

        auto gather_elements = std::make_shared<op::v6::GatherElements>(data, indices, 2);

        auto relu = std::make_shared<op::v0::Relu>(gather_elements);
        auto result = std::make_shared<ov::op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data, indices});
    }
}

TEST_F(TransformationTestsF, EliminateStridedSlice) {
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32,
                                                         PartialShape{ov::Dimension(), 4, ov::Dimension(), 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const =
            std::make_shared<op::v0::Constant>(ov::element::i64,
                                               ov::Shape{2},
                                               std::vector<int64_t>{0, std::numeric_limits<int64_t>::max()});
        auto optional_stride_const =
            std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        optional_stride_const,
                                                                        std::vector<int64_t>{1, 0, 1, 1},
                                                                        std::vector<int64_t>{1, 0, 1, 1});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::EliminateStridedSlice>();
    }
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32,
                                                         PartialShape{ov::Dimension(), 4, ov::Dimension(), 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto result = std::make_shared<op::v0::Result>(relu);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateStridedSlice_int32max) {
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, PartialShape{-1, 4, -1, 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const =
            std::make_shared<op::v0::Constant>(ov::element::i32,
                                               ov::Shape{2},
                                               std::vector<int64_t>{0, std::numeric_limits<int32_t>::max()});
        auto optional_stride_const =
            std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        optional_stride_const,
                                                                        std::vector<int64_t>{1, 0, 1, 1},
                                                                        std::vector<int64_t>{1, 0, 1, 1});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::EliminateStridedSlice>();
    }
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32,
                                                         PartialShape{ov::Dimension(), 4, ov::Dimension(), 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto result = std::make_shared<op::v0::Result>(relu);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateStridedSliceWithoutStrides) {
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32,
                                                         PartialShape{ov::Dimension(), 4, ov::Dimension(), 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const =
            std::make_shared<op::v0::Constant>(ov::element::i64,
                                               ov::Shape{2},
                                               std::vector<int64_t>{0, std::numeric_limits<int64_t>::max()});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        std::vector<int64_t>{1, 0},
                                                                        std::vector<int64_t>{1, 0});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::EliminateStridedSlice>();
    }
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32,
                                                         PartialShape{ov::Dimension(), 4, ov::Dimension(), 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto result = std::make_shared<op::v0::Result>(relu);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateStridedSliceByShape) {
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 4, 8, 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const =
            std::make_shared<op::v0::Constant>(ov::element::i64,
                                               ov::Shape{2},
                                               std::vector<int64_t>{0, std::numeric_limits<int64_t>::max()});
        auto optional_stride_const =
            std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        optional_stride_const,
                                                                        std::vector<int64_t>{1, 0, 1, 1},
                                                                        std::vector<int64_t>{1, 0, 1, 1});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::EliminateStridedSliceByShape>();
    }
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 4, 8, 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto result = std::make_shared<op::v0::Result>(relu);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateStridedSliceByShapeNegative) {
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 4, 8, 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int>{0, -1});
        auto optional_stride_const =
            std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        optional_stride_const,
                                                                        std::vector<int64_t>{1, 0},
                                                                        std::vector<int64_t>{1, 0});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::EliminateStridedSliceByShape>();
    }
    {
        auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 4, 8, 64});
        auto relu = std::make_shared<op::v0::Relu>(input);
        auto begin_const = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{0, 0});
        auto end_const = std::make_shared<op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int>{0, -1});
        auto optional_stride_const =
            std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int>{1, 1});
        auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                                        begin_const,
                                                                        end_const,
                                                                        optional_stride_const,
                                                                        std::vector<int64_t>{1, 0},
                                                                        std::vector<int64_t>{1, 0});
        auto result = std::make_shared<op::v0::Result>(strided_slice);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SqueezeBinaryReshape) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});

        auto axis = op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto squeeze = std::make_shared<op::v0::Squeeze>(data, axis);

        auto binary =
            std::make_shared<op::v1::Multiply>(squeeze, op::v0::Constant::create(element::f32, Shape{}, {0.2}));

        auto reshape =
            std::make_shared<op::v1::Reshape>(binary, op::v0::Constant::create(element::i32, Shape{1}, {1}), false);

        auto relu = std::make_shared<op::v0::Relu>(reshape);
        model = std::make_shared<ov::Model>(OutputVector{relu}, ParameterVector{data});
        manager.register_pass<ov::pass::NopElimination>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1});
        auto binary = std::make_shared<op::v1::Multiply>(data, op::v0::Constant::create(element::f32, Shape{1}, {0.2}));
        auto relu = std::make_shared<op::v0::Relu>(binary);
        model_ref = std::make_shared<ov::Model>(OutputVector{relu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EliminateSlice) {
    using namespace op::v0;
    auto type = element::i64;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape{3, 6, 9});
        auto slice = std::make_shared<op::v8::Slice>(input,
                                                     Constant::create(type, {1}, {0}),
                                                     Constant::create(type, {1}, {std::numeric_limits<int64_t>::max()}),
                                                     Constant::create(type, {1}, {1}),
                                                     Constant::create(type, {1}, {1}));
        auto relu = std::make_shared<Relu>(slice);

        auto result = std::make_shared<Result>(relu);

        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<pass::EliminateSlice>();
    }
    {
        auto input = std::make_shared<Parameter>(ov::element::f32, PartialShape{3, 6, 9});
        auto relu = std::make_shared<Relu>(input);
        auto result = std::make_shared<Result>(relu);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateSlice_int32max) {
    using namespace op::v0;
    auto type = element::i32;
    {
        auto input = std::make_shared<Parameter>(element::f32, PartialShape{3, 6, 9});
        auto slice = std::make_shared<op::v8::Slice>(input,
                                                     Constant::create(type, {1}, {0}),
                                                     Constant::create(type, {1}, {std::numeric_limits<int32_t>::max()}),
                                                     Constant::create(type, {1}, {1}),
                                                     Constant::create(type, {1}, {1}));
        auto relu = std::make_shared<Relu>(slice);

        auto result = std::make_shared<Result>(relu);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<pass::EliminateSlice>();
    }
    {
        auto input = std::make_shared<Parameter>(ov::element::f32, PartialShape{3, 6, 9});
        auto relu = std::make_shared<Relu>(input);
        auto result = std::make_shared<Result>(relu);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, TransposeWithEmptyOrder) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto empty_order = std::make_shared<op::v0::Constant>(element::i32, Shape{0}, std::vector<size_t>());
        auto transpose = std::make_shared<op::v1::Transpose>(relu, empty_order);

        auto result = std::make_shared<op::v0::Result>(transpose);
        model = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::NopElimination>();
    }
}

TEST_F(TransformationTestsF, TransposeElimination) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto order = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{0, 1});
        auto transpose = std::make_shared<op::v1::Transpose>(relu, order);

        auto result = std::make_shared<op::v0::Result>(transpose);
        model = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::NopElimination>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto result = std::make_shared<op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ScatterNDUpdates15Elimination) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{100, 256, 10, 15});
        auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{25, 0, 3});
        auto updates = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{25, 0, 15});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto scatter = std::make_shared<op::v15::ScatterNDUpdate>(relu, indices, updates);

        auto result = std::make_shared<op::v0::Result>(scatter);
        model = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data, indices, updates});
        manager.register_pass<ov::pass::EliminateScatterUpdate>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{100, 256, 10, 15});
        auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{25, 0, 3});
        auto updates = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{25, 0, 15});
        auto relu = std::make_shared<op::v0::Relu>(data);
        auto result = std::make_shared<op::v0::Result>(relu);
        model_ref = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data, indices, updates});
    }
}
