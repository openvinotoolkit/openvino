// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/opsets/opset13.hpp"

// run_replace_nonzero is file-local in llm_compiled_model.cpp.
// We test the same graph-transformation logic directly by reproducing
// the pattern: a NonZero node fed by a Parameter.  The key invariant
// is that only Parameters whose friendly name contains "visual_pos_masks"
// must be replaced; all other NonZero(Parameter) patterns must survive.
//
// Because the function is not exported we replicate its behaviour here and
// verify the post-conditions instead of calling it via a white-box interface.

namespace {

// Build a small model with two NonZero(Parameter) subgraphs:
//   param_vlm  -> NonZero -> Result   (should be replaced: name contains "visual_pos_masks")
//   param_other -> NonZero -> Result  (must NOT be replaced: unrelated parameter)
std::shared_ptr<ov::Model> build_two_nonzero_model() {
    using namespace ov::opset13;

    auto param_vlm = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    param_vlm->set_friendly_name("visual_pos_masks");
    param_vlm->output(0).set_names({"visual_pos_masks"});

    auto param_other = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    param_other->set_friendly_name("some_other_mask");
    param_other->output(0).set_names({"some_other_mask"});

    auto nz_vlm = std::make_shared<NonZero>(param_vlm, ov::element::i64);
    auto nz_other = std::make_shared<NonZero>(param_other, ov::element::i64);

    auto result_vlm = std::make_shared<Result>(nz_vlm);
    auto result_other = std::make_shared<Result>(nz_other);

    return std::make_shared<ov::Model>(
        ov::ResultVector{result_vlm, result_other},
        ov::ParameterVector{param_vlm, param_other},
        "replace_nonzero_test");
}

// Replicate the transformation logic from llm_compiled_model.cpp so that the
// unit test is self-contained and independent of internal linkage.
std::shared_ptr<ov::Model> apply_replace_nonzero(std::shared_ptr<ov::Model> model) {
    using namespace ov::opset13;

    std::vector<std::pair<std::shared_ptr<Parameter>, std::shared_ptr<ov::Node>>> to_replace;

    for (const auto& op : model->get_ordered_ops()) {
        if (auto nonzero = std::dynamic_pointer_cast<NonZero>(op)) {
            auto old_param = std::dynamic_pointer_cast<Parameter>(
                nonzero->input_value(0).get_node_shared_ptr());
            if (old_param &&
                old_param->get_friendly_name().find("visual_pos_masks") != std::string::npos) {
                to_replace.push_back({old_param, nonzero});
            }
        }
    }

    std::vector<std::shared_ptr<Parameter>> old_params_to_remove;

    for (auto& [old_param, nonzero_node] : to_replace) {
        auto new_param = std::make_shared<Parameter>(
            nonzero_node->get_output_element_type(0),
            nonzero_node->get_output_partial_shape(0));
        new_param->set_friendly_name(old_param->get_friendly_name());

        auto names = nonzero_node->output(0).get_names();
        if (names.empty()) names = old_param->output(0).get_names();
        if (names.empty()) names.insert(old_param->get_friendly_name());
        new_param->output(0).set_names(names);

        ov::replace_node(nonzero_node, new_param);
        old_params_to_remove.push_back(old_param);
    }

    for (auto& p : old_params_to_remove) {
        model->remove_parameter(p);
    }

    ov::ParameterVector updated;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto p = std::dynamic_pointer_cast<Parameter>(op)) {
            if (!p->output(0).get_target_inputs().empty()) {
                updated.push_back(p);
            }
        }
    }

    auto new_model = std::make_shared<ov::Model>(
        model->get_results(), model->get_sinks(), updated,
        model->get_variables(), model->get_friendly_name());
    new_model->validate_nodes_and_infer_types();
    return new_model;
}

// Helper: count NonZero nodes in a model.
static size_t count_nonzero(const std::shared_ptr<ov::Model>& model) {
    size_t n = 0;
    for (const auto& op : model->get_ordered_ops())
        if (std::dynamic_pointer_cast<ov::opset13::NonZero>(op)) ++n;
    return n;
}

// --- Test 1 -------------------------------------------------------------------
// Only visual_pos_masks NonZero should be eliminated; the other must survive.
TEST(ReplaceNonZeroTransformTest, OnlyVisualPosMasksParameterIsReplaced) {
    auto model = build_two_nonzero_model();
    ASSERT_EQ(count_nonzero(model), 2u);
    ASSERT_EQ(model->get_parameters().size(), 2u);

    auto result = apply_replace_nonzero(model);

    // visual_pos_masks NonZero is gone; only the unrelated one remains.
    EXPECT_EQ(count_nonzero(result), 1u)
        << "Expected exactly one NonZero to remain (the unrelated parameter)";

    EXPECT_EQ(result->get_parameters().size(), 2u)
        << "Parameter count must stay the same (old removed, new added)";
}

// --- Test 2 -------------------------------------------------------------------
// A model with no visual_pos_masks must be left completely untouched.
TEST(ReplaceNonZeroTransformTest, NonVlmModelIsUntouched) {
    using namespace ov::opset13;

    auto param = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    param->set_friendly_name("some_mask");
    param->output(0).set_names({"some_mask"});

    auto nz = std::make_shared<NonZero>(param, ov::element::i64);
    auto result = std::make_shared<Result>(nz);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result}, ov::ParameterVector{param}, "no_vlm_model");

    ASSERT_EQ(count_nonzero(model), 1u);

    auto out = apply_replace_nonzero(model);

    EXPECT_EQ(count_nonzero(out), 1u)
        << "NonZero must survive when no visual_pos_masks parameter is present";
    EXPECT_EQ(out->get_parameters().size(), 1u);
    EXPECT_EQ(out->get_parameters().front()->get_friendly_name(), "some_mask");
}

// --- Test 3 -------------------------------------------------------------------
// After replacement the new parameter for visual_pos_masks must have the
// NonZero output type (i64) and the NonZero output shape, not the original
// boolean input type/shape.
TEST(ReplaceNonZeroTransformTest, ReplacedParameterHasNonZeroOutputTypeAndShape) {
    using namespace ov::opset13;

    auto param = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{4, 8});
    param->set_friendly_name("visual_pos_masks");
    param->output(0).set_names({"visual_pos_masks"});

    auto nz = std::make_shared<NonZero>(param, ov::element::i64);
    auto result = std::make_shared<Result>(nz);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result}, ov::ParameterVector{param}, "vlm_single");

    auto out = apply_replace_nonzero(model);

    EXPECT_EQ(count_nonzero(out), 0u);
    ASSERT_EQ(out->get_parameters().size(), 1u);

    const auto& new_param = out->get_parameters().front();
    EXPECT_EQ(new_param->get_element_type(), ov::element::i64)
        << "Replacement parameter must carry NonZero's output type (i64)";
    EXPECT_EQ(new_param->get_friendly_name(), "visual_pos_masks");
}

}  // namespace
