// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_precision.hpp>

#include "plugin/transformations/disable_fp16_comp_rms.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static const std::string name_rms_1 = "rms_1";
static const std::string name_rms_2 = "rms_2";

// This model creates the exact pattern that DisableFP16CompForGemma3RMSPattern is looking for.
// (Add, RMS) -> Add -> RMS
static std::shared_ptr<ov::Model> create_model_to_match(bool use_convert = false) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    // Pattern part 1: add_m
    auto add_m = std::make_shared<ov::op::v1::Add>(input1, input2);

    // Pattern part 2: rms_post_m
    std::shared_ptr<ov::Node> rms_const_or_convert_1;
    if (use_convert) {
        auto const_node_1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_1 = std::make_shared<ov::op::v0::Convert>(const_node_1, ov::element::f32);
    } else {
        rms_const_or_convert_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_post_m = std::make_shared<ov::op::internal::RMS>(input3, rms_const_or_convert_1, 1e-5);
    rms_post_m->set_friendly_name(name_rms_1);

    // Pattern part 3: add_1_m
    auto add_1_m = std::make_shared<ov::op::v1::Add>(add_m, rms_post_m);

    // Pattern part 4: rms_m
    std::shared_ptr<ov::Node> rms_const_or_convert_2;
    if (use_convert) {
        auto const_node_2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_2 = std::make_shared<ov::op::v0::Convert>(const_node_2, ov::element::f32);
    } else {
        rms_const_or_convert_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_m = std::make_shared<ov::op::internal::RMS>(add_1_m, rms_const_or_convert_2, 1e-5);
    rms_m->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_m}, ov::ParameterVector{input1, input2, input3});
}

// This model has a similar structure but doesn't match the specific pattern.
static std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    auto rms_const_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_1 = std::make_shared<ov::op::internal::RMS>(input1, rms_const_1, 1e-5);
    rms_1->set_friendly_name(name_rms_1);

    auto some_other_op = std::make_shared<ov::op::v1::Add>(rms_1, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}));

    auto rms_const_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_2 = std::make_shared<ov::op::internal::RMS>(some_other_op, rms_const_2, 1e-5);
    rms_2->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_2}, ov::ParameterVector{input1});
}

static void run_test(std::shared_ptr<ov::Model> model,
                     const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGemma3RMSPattern>();

    precisions_map fp_convert_precision_map = {
        {ov::element::f32, ov::element::f16}
    };
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::fp16_compression_is_disabled(op))
                    << "FP16 compression is not disabled for node: " << op->get_friendly_name();
            } else {
                ASSERT_FALSE(ov::fp16_compression_is_disabled(op))
                    << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
            }
        }
    }
}

TEST(TransformationTests, DisableFP16CompForRMS_Positive) {
    auto model = create_model_to_match();
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, true},
        {name_rms_2, true}
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_PositiveConvert) {
    auto model = create_model_to_match(true);
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, true},
        {name_rms_2, true}
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_Negative) {
    auto model = create_model_not_to_match();
    // In the non-matching model, no RMS node should have FP16 compression disabled by the pass.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, false},
        {name_rms_2, false}
    };
    run_test(model, expected_status);
}
