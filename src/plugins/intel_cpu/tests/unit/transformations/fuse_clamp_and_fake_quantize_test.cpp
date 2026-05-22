// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/pass/fuse_clamp_and_fake_quantize.hpp"
#include "transformations/init_node_info.hpp"
#include "utils/rt_info/fake_quantize_clamp_bounds.hpp"

namespace {

std::shared_ptr<ov::op::v0::FakeQuantize> find_fake_quantize(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
            return fq;
        }
    }

    return nullptr;
}

std::size_t count_clamps(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Clamp>(node)) {
            count++;
        }
    }

    return count;
}

std::shared_ptr<ov::Model> create_model(const std::vector<std::pair<float, float>>& clamp_ranges) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8});

    std::shared_ptr<ov::Node> data = input;
    for (const auto& [min_value, max_value] : clamp_ranges) {
        data = std::make_shared<ov::op::v0::Clamp>(data, min_value, max_value);
    }

    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {4.f});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.f});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {255.f});

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, 256);
    auto result = std::make_shared<ov::op::v0::Result>(fq);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

}  // namespace

TEST(TransformationTests, FuseClampAndFakeQuantize_RemovesClampBeforeFakeQuantize) {
    auto model = create_model({{0.f, 2.f}});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 0);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(fq->get_input_node_shared_ptr(0)));

    const auto bounds = ov::intel_cpu::get_fake_quantize_clamp_bounds(fq);
    ASSERT_TRUE(bounds.has_value());
    EXPECT_FLOAT_EQ(bounds->low(), 0.f);
    EXPECT_FLOAT_EQ(bounds->high(), 2.f);
}

TEST(TransformationTests, FuseClampAndFakeQuantize_MergesNestedClamps) {
    auto model = create_model({{-1.f, 3.f}, {0.f, 2.f}});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 0);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(fq->get_input_node_shared_ptr(0)));

    const auto bounds = ov::intel_cpu::get_fake_quantize_clamp_bounds(fq);
    ASSERT_TRUE(bounds.has_value());
    EXPECT_FLOAT_EQ(bounds->low(), 0.f);
    EXPECT_FLOAT_EQ(bounds->high(), 2.f);
}