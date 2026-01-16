// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_precision.hpp>

#include "plugin/transformations/disable_fp16_comp_gather_if.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/add.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static const std::string name_if = "prim::If/If";
static const std::string name_gather = "__module.model.roi_heads.mask_pooler/aten::select/Gather";

// This model creates the exact pattern that DisableFP16CompForDetectron2MaskRCNNGatherIfPattern is looking for.
static std::shared_ptr<ov::Model> create_model_to_match(bool use_convert = false) {

    // Gather_ND -> If
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4});
    auto pred = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto indices_nd = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 1});
    auto gather_nd = std::make_shared<ov::op::v8::GatherND>(data, indices_nd, 0);
    auto gather_nd_out_shape = gather_nd->get_output_shape(0);

    auto then_gather = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, gather_nd->get_output_partial_shape(0));
    auto then_const = ov::op::v0::Constant::create(ov::element::f16,
        gather_nd_out_shape, std::vector<float>(ov::shape_size(gather_nd_out_shape), 1.0f));
    auto then_add = std::make_shared<ov::op::v1::Add>(then_gather, then_const);
    auto then_result = std::make_shared<ov::op::v0::Result>(then_add);
    auto then_body = std::make_shared<ov::Model>(
        ov::ResultVector{then_result},
        ov::ParameterVector{then_gather});

    auto else_gather = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, gather_nd->get_output_partial_shape(0));
    auto else_const = ov::op::v0::Constant::create(ov::element::f16,
        gather_nd_out_shape, std::vector<float>(ov::shape_size(gather_nd_out_shape), 2.0f));
    auto else_add = std::make_shared<ov::op::v1::Add>(else_gather, else_const);
    auto else_result = std::make_shared<ov::op::v0::Result>(else_add);
    auto else_body = std::make_shared<ov::Model>(
        ov::ResultVector{else_result},
        ov::ParameterVector{else_gather});

    auto if_node = std::make_shared<ov::op::v8::If>(pred);
    if_node->set_then_body(then_body);
    if_node->set_else_body(else_body);
    if_node->set_friendly_name(name_if);

    if_node->set_input(gather_nd->output(0), then_gather, else_gather);
    if_node->set_output(then_result, else_result);

    auto if_result = std::make_shared<ov::op::v0::Result>(if_node->output(0));

    // Gather -> Gather
    auto indices0  = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    auto indices1  = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto gather0_node = std::make_shared<ov::op::v8::Gather>(data, indices0, axis0);
    auto gather1_node = std::make_shared<ov::op::v8::Gather>(gather0_node, indices1, axis1);
    gather1_node->set_friendly_name(name_gather);

    auto gather_result = std::make_shared<ov::op::v0::Result>(gather1_node);

    return std::make_shared<ov::Model>(
        ov::ResultVector{if_result, gather_result},
        ov::ParameterVector{pred, data, indices_nd, indices0, indices1});
}

static void run_test(std::shared_ptr<ov::Model> model,
                     const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForDetectron2MaskRCNNGatherIfPattern>();

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

TEST(TransformationTests, DisableFP16CompForDetectron2MaskRCNNGatherIf) {
    auto model = create_model_to_match();
    // In the matching pattern, 'If' and 'Gather' should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_if, true},
        {name_gather, true},
    };
    run_test(model, expected_status);
}
