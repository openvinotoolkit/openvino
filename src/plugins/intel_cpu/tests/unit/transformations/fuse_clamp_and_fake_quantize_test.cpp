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

std::shared_ptr<ov::Model> create_model(const std::vector<std::pair<float, float>>& clamp_ranges,
                                        const std::vector<float>& input_low_values = {1.f},
                                        const std::vector<float>& input_high_values = {4.f},
                                        const std::vector<float>& output_low_values = {0.f},
                                        const std::vector<float>& output_high_values = {255.f},
                                        const size_t levels = 256) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8});

    std::shared_ptr<ov::Node> data = input;
    for (const auto& [min_value, max_value] : clamp_ranges) {
        data = std::make_shared<ov::op::v0::Clamp>(data, min_value, max_value);
    }

    auto input_low =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{input_low_values.size()}, input_low_values);
    auto input_high =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{input_high_values.size()}, input_high_values);
    auto output_low =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{output_low_values.size()}, output_low_values);
    auto output_high =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{output_high_values.size()}, output_high_values);

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    auto result = std::make_shared<ov::op::v0::Result>(fq);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

}  // namespace

TEST(TransformationTests, FuseClampAndFakeQuantize_RemovesRedundantClampBeforeFakeQuantize) {
    auto model = create_model({{0.f, 10.f}});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 0);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(fq->get_input_node_shared_ptr(0)));
}

TEST(TransformationTests, FuseClampAndFakeQuantize_KeepsClampThatNarrowsFakeQuantizeInputRange) {
    auto model = create_model({{0.f, 2.f}});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 1);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Clamp>(fq->get_input_node_shared_ptr(0)));
}

TEST(TransformationTests, FuseClampAndFakeQuantize_RemovesRedundantClampBeforeLevels2FakeQuantize) {
    auto model = create_model({{0.f, 2.f}}, {1.f}, {1.f}, {0.f}, {1.f}, 2);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 0);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(fq->get_input_node_shared_ptr(0)));
}

TEST(TransformationTests, FuseClampAndFakeQuantize_RemovesConsecutiveRedundantClamps) {
    auto model = create_model({{0.f, 8.f}, {-1.f, 10.f}});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_cpu::FuseClampAndFakeQuantize>();
    manager.run_passes(model);

    const auto fq = find_fake_quantize(model);
    ASSERT_NE(fq, nullptr);
    EXPECT_EQ(count_clamps(model), 0);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(fq->get_input_node_shared_ptr(0)));
}