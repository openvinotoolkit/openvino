// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"

#include <intel_gpu/op/atan2.hpp>
#include <plugin/transformations/fuse_atan2_decomposed.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

// Constructs the post-ConvertDivide form of the frontend atan2 decomposition,
// matching what FuseAtan2Decomposed sees in the GPU pipeline.
struct Atan2DecomposedModel {
    std::shared_ptr<ov::op::v0::Parameter> y;  // lhs (imag)
    std::shared_ptr<ov::op::v0::Parameter> x;  // rhs (real)
    std::shared_ptr<ov::op::v1::Select> root;  // Sel3
    std::shared_ptr<ov::Model> model;
};

Atan2DecomposedModel build_decomposed_atan2(ov::element::Type et, bool use_divide) {
    Atan2DecomposedModel m;
    m.y = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, -1});
    m.x = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, -1});

    std::shared_ptr<ov::Node> div_out;
    if (use_divide) {
        div_out = std::make_shared<ov::op::v1::Divide>(m.y, m.x);
    } else {
        auto neg_one = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-1.0f});
        auto pow = std::make_shared<ov::op::v1::Power>(m.x, neg_one);
        div_out = std::make_shared<ov::op::v1::Multiply>(m.y, pow);
    }
    auto atan = std::make_shared<ov::op::v0::Atan>(div_out);

    auto pi = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{3.14159265f});
    auto neg_pi = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-3.14159265f});
    auto half_pi = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{1.57079633f});
    auto neg_half_pi = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-1.57079633f});
    auto zero = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{0.0f});

    auto atan_plus_pi = std::make_shared<ov::op::v1::Add>(atan, pi);
    auto atan_minus_pi = std::make_shared<ov::op::v1::Add>(atan, neg_pi);

    auto x_lt = std::make_shared<ov::op::v1::Less>(m.x, zero);
    auto y_ge = std::make_shared<ov::op::v1::GreaterEqual>(m.y, zero);
    auto add_pi_cond = std::make_shared<ov::op::v1::LogicalAnd>(x_lt, y_ge);
    auto sel1 = std::make_shared<ov::op::v1::Select>(add_pi_cond, atan_plus_pi, atan_minus_pi);

    auto x_gt = std::make_shared<ov::op::v1::Greater>(m.x, zero);
    auto sel2 = std::make_shared<ov::op::v1::Select>(x_gt, atan, sel1);

    auto x_eq = std::make_shared<ov::op::v1::Equal>(m.x, zero);
    auto y_gt = std::make_shared<ov::op::v1::Greater>(m.y, zero);
    auto y_lt = std::make_shared<ov::op::v1::Less>(m.y, zero);
    auto half_pi_cond = std::make_shared<ov::op::v1::LogicalAnd>(x_eq, y_gt);
    auto neg_half_pi_cond = std::make_shared<ov::op::v1::LogicalAnd>(x_eq, y_lt);
    auto special_cond = std::make_shared<ov::op::v1::LogicalOr>(half_pi_cond, neg_half_pi_cond);
    auto special_select = std::make_shared<ov::op::v1::Select>(half_pi_cond, half_pi, neg_half_pi);

    m.root = std::make_shared<ov::op::v1::Select>(special_cond, special_select, sel2);
    m.model = std::make_shared<ov::Model>(ov::OutputVector{m.root}, ov::ParameterVector{m.y, m.x});
    return m;
}

bool model_contains_atan2(const std::shared_ptr<ov::Model>& m) {
    for (const auto& node : m->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::intel_gpu::op::Atan2>(node))
            return true;
    }
    return false;
}

}  // namespace

TEST(FuseAtan2DecomposedTest, FusesPostConvertDivideForm) {
    auto m = build_decomposed_atan2(ov::element::f16, /*use_divide=*/false);

    ov::pass::Manager manager;
    manager.register_pass<FuseAtan2Decomposed>();
    manager.run_passes(m.model);

    ASSERT_TRUE(model_contains_atan2(m.model)) << "Atan2 op should be inserted";

    // Result should now consume the Atan2 directly.
    ASSERT_EQ(m.model->get_results().size(), 1u);
    auto result_input = m.model->get_results()[0]->get_input_node_shared_ptr(0);
    auto atan2 = ov::as_type_ptr<ov::intel_gpu::op::Atan2>(result_input);
    ASSERT_TRUE(atan2 != nullptr) << "Result must be fed by Atan2";

    // input0 = y, input1 = x.
    EXPECT_EQ(atan2->get_input_node_shared_ptr(0).get(), m.y.get());
    EXPECT_EQ(atan2->get_input_node_shared_ptr(1).get(), m.x.get());
}

TEST(FuseAtan2DecomposedTest, FusesDivideForm) {
    auto m = build_decomposed_atan2(ov::element::f32, /*use_divide=*/true);

    ov::pass::Manager manager;
    manager.register_pass<FuseAtan2Decomposed>();
    manager.run_passes(m.model);

    ASSERT_TRUE(model_contains_atan2(m.model));
}

TEST(FuseAtan2DecomposedTest, DoesNotFuseUnrelatedSelect) {
    // A bare Select(cond, x, y) without the surrounding atan structure must
    // not be touched.
    auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1});
    auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto sel = std::make_shared<ov::op::v1::Select>(cond, a, b);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sel}, ov::ParameterVector{cond, a, b});

    ov::pass::Manager manager;
    manager.register_pass<FuseAtan2Decomposed>();
    manager.run_passes(model);

    EXPECT_FALSE(model_contains_atan2(model));
}
