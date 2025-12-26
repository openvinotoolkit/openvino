// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_unroll_patterns.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#define MATCHER_SCOPE(region) const std::string matcher_name = std::string(typeid(*this).name()) + "_" + #region

namespace ov {
namespace npuw {
namespace pass {

// =============================================================================
// UnrollBatchedMatMul
// =============================================================================
// Matches MoE expert MatMul pattern and unrolls batched dimension to N branches
// Pattern: input_param → convert → tile → reshape ─────────────┐
//          scale_param + weights_param → multiply → convert ───┤→ MatMul
//
// Transforms to N expert branches with Concat output

UnrollBatchedMatMul::UnrollBatchedMatMul(size_t num_experts, std::shared_ptr<ov::Model> model)
    : num_experts_(num_experts),
      model_(model) {
    MATCHER_SCOPE(UnrollBatchedMatMul);

    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::opset1::MatMul>();

    auto callback = [this](ov::pass::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ov::opset1::MatMul>(m.get_match_root());
        if (!matmul)
            return false;

        LOG_INFO("UnrollBatchedMatMul: Checking MatMul " << matmul->get_friendly_name());
        std::cout << "UnrollBatchedMatMul: Checking MatMul " << matmul->get_friendly_name() << std::endl;

        auto matmul_input0 = matmul->input_value(0);
        auto matmul_input1 = matmul->input_value(1);

        LOG_DEBUG("  Input0 type: " << matmul_input0.get_node_shared_ptr()->get_type_name());
        LOG_DEBUG("  Input1 type: " << matmul_input1.get_node_shared_ptr()->get_type_name());

        // Check input0: Reshape (possibly through Squeeze/Convert)
        auto input0_node = matmul_input0.get_node_shared_ptr();
        auto reshape_node = std::dynamic_pointer_cast<ov::opset1::Reshape>(input0_node);

        if (!reshape_node) {
            if (auto squeeze = std::dynamic_pointer_cast<ov::opset1::Squeeze>(input0_node)) {
                reshape_node =
                    std::dynamic_pointer_cast<ov::opset1::Reshape>(squeeze->input_value(0).get_node_shared_ptr());
            } else if (auto convert = std::dynamic_pointer_cast<ov::opset1::Convert>(input0_node)) {
                reshape_node =
                    std::dynamic_pointer_cast<ov::opset1::Reshape>(convert->input_value(0).get_node_shared_ptr());
            }
        }

        if (!reshape_node) {
            LOG_DEBUG("  Input0 is not Reshape, skipping");
            std::cout << "  Input0 is not Reshape, skipping" << std::endl;
            std::cout << " Input0 type: " << matmul_input0.get_node_shared_ptr()->get_type_name() << std::endl;
            return false;
        }

        // Check Reshape → Tile → Convert chain
        auto tile_node =
            std::dynamic_pointer_cast<ov::opset1::Tile>(reshape_node->input_value(0).get_node_shared_ptr());
        if (!tile_node) {
            LOG_DEBUG("  Reshape input is not Tile, skipping");
            std::cout << "  Reshape input is not Tile, skipping" << std::endl;
            return false;
        }

        auto convert_input_node =
            std::dynamic_pointer_cast<ov::opset1::Convert>(tile_node->input_value(0).get_node_shared_ptr());
        if (!convert_input_node) {
            LOG_DEBUG("  Tile input is not Convert, skipping");
            std::cout << "  Tile input is not Convert, skipping" << std::endl;
            return false;
        }

        // Check input1: Multiply (possibly through Convert)
        auto input1_node = matmul_input1.get_node_shared_ptr();
        std::shared_ptr<ov::opset1::Convert> convert_after_multiply;
        std::shared_ptr<ov::opset1::Multiply> multiply_weights_node;

        if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(input1_node)) {
            convert_after_multiply = conv;
            multiply_weights_node =
                std::dynamic_pointer_cast<ov::opset1::Multiply>(conv->input_value(0).get_node_shared_ptr());
        } else {
            multiply_weights_node = std::dynamic_pointer_cast<ov::opset1::Multiply>(input1_node);
        }

        if (!multiply_weights_node) {
            LOG_DEBUG("  Input1 is not Multiply, skipping");
            std::cout << "  Input1 is not Multiply, skipping" << std::endl;
            return false;
        }

        auto input_param_source = convert_input_node->input_value(0);
        auto multiply_input0 = multiply_weights_node->input_value(0);
        auto multiply_input1 = multiply_weights_node->input_value(1);

        // Helper: Skip Convert operation to access underlying Parameter node
        auto skip_convert = [](ov::Output<ov::Node> out) -> ov::Output<ov::Node> {
            if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(out.get_node_shared_ptr())) {
                return conv->input_value(0);
            }
            return out;
        };

        // Helper: Calculate total number of elements in a shape for size comparison
        auto calc_total_size = [](const ov::PartialShape& shape) -> size_t {
            if (!shape.rank().is_static())
                return 0;
            size_t total = 1;
            for (int64_t i = 0; i < shape.rank().get_length(); ++i) {
                if (shape[i].is_static()) {
                    total *= shape[i].get_length();
                }
            }
            return total;
        };

        auto mult_in0_skip_convert = skip_convert(multiply_input0);
        auto mult_in1_skip_convert = skip_convert(multiply_input1);

        size_t size0 = calc_total_size(mult_in0_skip_convert.get_partial_shape());
        size_t size1 = calc_total_size(mult_in1_skip_convert.get_partial_shape());

        // Determine which input is scale vs weights by comparing total element count
        // Convention: larger parameter is weights matrix, smaller is scale vector
        ov::Output<ov::Node> scale_param_source, weights_param_source;
        if (size0 > size1) {
            weights_param_source = multiply_input0;
            scale_param_source = multiply_input1;
        } else {
            weights_param_source = multiply_input1;
            scale_param_source = multiply_input0;
        }

        LOG_INFO("  Found pattern: input_param → convert → tile → reshape");
        LOG_INFO("                  scale_param + weights_param → multiply → MatMul");
        LOG_INFO("  Creating " << num_experts_ << " expert branches...");
        std::cout << "  Found pattern: input_param → convert → tile → reshape" << std::endl;
        std::cout << "                  scale_param + weights_param → multiply → MatMul" << std::endl;
        std::cout << "  Creating " << num_experts_ << " expert branches..." << std::endl;

        // Helper: Extract Parameter node, skipping intermediate Convert if present
        auto get_param_node = [](ov::Output<ov::Node> out) -> std::shared_ptr<ov::op::v0::Parameter> {
            auto node = out.get_node_shared_ptr();
            if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(node)) {
                node = conv->input_value(0).get_node_shared_ptr();
            }
            return std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        };

        auto scale_param = get_param_node(scale_param_source);
        auto weights_param = get_param_node(weights_param_source);

        if (!scale_param || !weights_param) {
            LOG_ERROR("  Could not find parameter nodes");
            return false;
        }

        // Calculate per-expert parameter shapes by splitting expert dimension
        // Original: [num_experts, dim1, dim2, ...] → Per-expert: [1, dim1, dim2, ...]
        // This splits batched parameters into individual expert parameters
        auto scale_orig_shape = scale_param->get_partial_shape().to_shape();
        auto weights_orig_shape = weights_param->get_partial_shape().to_shape();

        // Find which dimension contains num_experts (usually dimension 0)
        size_t expert_dim = 0;
        for (size_t i = 0; i < scale_orig_shape.size(); ++i) {
            if (scale_orig_shape[i] == num_experts_) {
                expert_dim = i;
                break;
            }
        }

        ov::Shape scale_new_shape = scale_orig_shape;
        ov::Shape weights_new_shape = weights_orig_shape;
        scale_new_shape[expert_dim] = 1;
        weights_new_shape[expert_dim] = 1;

        LOG_INFO("  Original scale shape: " << scale_orig_shape << " → " << scale_new_shape);
        LOG_INFO("  Original weights shape: " << weights_orig_shape << " → " << weights_new_shape);

        ov::NodeVector expert_outputs;
        ov::ParameterVector new_params;

        for (size_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
            // 1. Convert input
            auto new_convert =
                std::make_shared<ov::opset1::Convert>(input_param_source, convert_input_node->get_destination_type());
            new_convert->set_friendly_name(convert_input_node->get_friendly_name() + "/expert_" +
                                           std::to_string(expert_idx));

            // 2. Create per-expert Reshape with adjusted target shape
            // Only handle case where first dimension exactly equals num_experts
            // Original reshape: [num_experts, ...] → Per-expert reshape: [1, ...]
            auto orig_reshape_shape_const =
                std::dynamic_pointer_cast<ov::opset1::Constant>(reshape_node->input_value(1).get_node_shared_ptr());
            if (!orig_reshape_shape_const) {
                LOG_ERROR("  Reshape shape is not a Constant");
                return false;
            }

            auto orig_shape_vec = orig_reshape_shape_const->cast_vector<int64_t>();

            // Validate: first dimension must exactly equal num_experts
            if (orig_shape_vec.empty() || orig_shape_vec[0] != static_cast<int64_t>(num_experts_)) {
                LOG_DEBUG("  Reshape first dimension " << (orig_shape_vec.empty() ? 0 : orig_shape_vec[0])
                                                       << " does not match num_experts " << num_experts_
                                                       << ", skipping");
                return false;
            }

            std::vector<int64_t> new_shape_vec = orig_shape_vec;
            new_shape_vec[0] = 1;  // Change first dimension from num_experts to 1

            auto new_reshape_shape_const = std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                                                  ov::Shape{new_shape_vec.size()},
                                                                                  new_shape_vec);
            auto new_reshape =
                std::make_shared<ov::opset1::Reshape>(new_convert->output(0), new_reshape_shape_const, false);
            new_reshape->set_friendly_name(reshape_node->get_friendly_name() + "/expert_" + std::to_string(expert_idx));

            // 3. Create new scale parameter
            auto new_scale_param = std::make_shared<ov::op::v0::Parameter>(scale_param->get_element_type(),
                                                                           ov::PartialShape(scale_new_shape));
            new_scale_param->set_friendly_name(scale_param->get_friendly_name() + "/expert_" +
                                               std::to_string(expert_idx));
            new_params.push_back(new_scale_param);

            // 4. Create new weights parameter
            auto new_weights_param = std::make_shared<ov::op::v0::Parameter>(weights_param->get_element_type(),
                                                                             ov::PartialShape(weights_new_shape));
            new_weights_param->set_friendly_name(weights_param->get_friendly_name() + "/expert_" +
                                                 std::to_string(expert_idx));
            new_params.push_back(new_weights_param);

            // 5. Apply Convert to weights if needed
            ov::Output<ov::Node> weights_for_multiply;
            if (auto weights_convert =
                    std::dynamic_pointer_cast<ov::opset1::Convert>(weights_param_source.get_node_shared_ptr())) {
                auto new_weights_convert =
                    std::make_shared<ov::opset1::Convert>(new_weights_param, weights_convert->get_destination_type());
                new_weights_convert->set_friendly_name(weights_convert->get_friendly_name() + "/expert_" +
                                                       std::to_string(expert_idx));
                weights_for_multiply = new_weights_convert->output(0);
            } else {
                weights_for_multiply = new_weights_param->output(0);
            }

            // 6. Apply Convert to scale if needed
            ov::Output<ov::Node> scale_for_multiply;
            if (auto scale_convert =
                    std::dynamic_pointer_cast<ov::opset1::Convert>(scale_param_source.get_node_shared_ptr())) {
                auto new_scale_convert =
                    std::make_shared<ov::opset1::Convert>(new_scale_param, scale_convert->get_destination_type());
                new_scale_convert->set_friendly_name(scale_convert->get_friendly_name() + "/expert_" +
                                                     std::to_string(expert_idx));
                scale_for_multiply = new_scale_convert->output(0);
            } else {
                scale_for_multiply = new_scale_param->output(0);
            }

            // 7. Multiply: scale * weights
            auto new_multiply = std::make_shared<ov::opset1::Multiply>(scale_for_multiply, weights_for_multiply);
            new_multiply->set_friendly_name(multiply_weights_node->get_friendly_name() + "/expert_" +
                                            std::to_string(expert_idx));

            // 8. Convert after Multiply if needed
            ov::Output<ov::Node> weights_for_matmul;
            if (convert_after_multiply) {
                auto new_convert_after_multiply =
                    std::make_shared<ov::opset1::Convert>(new_multiply, convert_after_multiply->get_destination_type());
                new_convert_after_multiply->set_friendly_name(convert_after_multiply->get_friendly_name() + "/expert_" +
                                                              std::to_string(expert_idx));
                weights_for_matmul = new_convert_after_multiply->output(0);
            } else {
                weights_for_matmul = new_multiply->output(0);
            }

            // 9. MatMul
            auto new_matmul = std::make_shared<ov::opset1::MatMul>(new_reshape,
                                                                   weights_for_matmul,
                                                                   matmul->get_transpose_a(),
                                                                   matmul->get_transpose_b());
            new_matmul->set_friendly_name(matmul->get_friendly_name() + "/expert_" + std::to_string(expert_idx));

            expert_outputs.push_back(new_matmul);
        }

        // Concat all expert outputs
        auto concat = std::make_shared<ov::opset1::Concat>(expert_outputs, 0);
        concat->set_friendly_name(matmul->get_friendly_name() + "/concat");

        // Register new parameters with model
        model_->add_parameters(new_params);

        // Replace original MatMul
        ov::copy_runtime_info(matmul, concat);
        ov::replace_node(matmul, concat);

        LOG_INFO("  Successfully created " << num_experts_ << " expert branches with " << new_params.size()
                                           << " new parameters");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushElementwiseBeforeConcat
// =============================================================================
// Transforms: Parameter[N,...] + Concat([a,b,c])  =>  Concat([a+param[0], b+param[1], c+param[2]])
// Supports Add and Multiply operations
// Optimization: Distributes elementwise operations into concat branches to enable further optimizations

PushElementwiseBeforeConcat::PushElementwiseBeforeConcat(std::shared_ptr<ov::Model> model) : model_(model) {
    MATCHER_SCOPE(PushElementwiseBeforeConcat);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto elementwise_pattern = ov::pass::pattern::wrap_type<ov::opset1::Add, ov::opset1::Multiply>(
        {concat_pattern, ov::pass::pattern::any_input()});

    auto callback = [this, concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto elementwise_node = m.get_match_root();
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!elementwise_node || !concat)
            return false;

        // Check if it's Add or Multiply
        auto add_op = std::dynamic_pointer_cast<ov::opset1::Add>(elementwise_node);
        auto multiply_op = std::dynamic_pointer_cast<ov::opset1::Multiply>(elementwise_node);

        if (!add_op && !multiply_op)
            return false;

        // Find which input is concat and which is the parameter
        ov::Output<ov::Node> other_input;
        if (elementwise_node->input_value(0).get_node_shared_ptr() == concat) {
            other_input = elementwise_node->input_value(1);
        } else if (elementwise_node->input_value(1).get_node_shared_ptr() == concat) {
            other_input = elementwise_node->input_value(0);
        } else {
            return false;
        }

        // Check if other_input comes from a Parameter (possibly through Convert)
        auto other_node = other_input.get_node_shared_ptr();
        std::shared_ptr<ov::op::v0::Parameter> param_node;
        std::shared_ptr<ov::opset1::Convert> convert_node;

        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(other_node)) {
            param_node = param;
        } else if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(other_node)) {
            convert_node = conv;
            param_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(conv->input_value(0).get_node_shared_ptr());
        }

        if (!param_node) {
            LOG_DEBUG("  Other input is not a Parameter, skipping");
            return false;
        }

        // Check parameter shape: should be [N, ...] where N = concat input count
        auto param_shape = param_node->get_partial_shape();
        if (!param_shape.rank().is_static() || !param_shape[0].is_static()) {
            return false;
        }

        size_t num_branches = concat->get_input_size();
        if (param_shape[0].get_length() != static_cast<int64_t>(num_branches)) {
            return false;
        }

        std::string op_type = add_op ? "Add" : "Multiply";
        LOG_INFO("PushElementwiseBeforeConcat: Pushing " << op_type << " with Parameter before Concat");
        LOG_INFO("  Parameter shape: " << param_shape << ", num branches: " << num_branches);

        // Create new per-branch parameters
        auto orig_shape = param_shape.to_shape();
        ov::Shape new_param_shape = orig_shape;
        new_param_shape[0] = 1;  // Change first dimension from N to 1

        auto concat_inputs = concat->input_values();
        ov::NodeVector new_add_ops;
        ov::ParameterVector new_params;

        for (size_t i = 0; i < num_branches; ++i) {
            // Create new parameter for this branch
            auto new_param = std::make_shared<ov::op::v0::Parameter>(param_node->get_element_type(),
                                                                     ov::PartialShape(new_param_shape));
            new_param->set_friendly_name(param_node->get_friendly_name() + "/branch_" + std::to_string(i));
            new_params.push_back(new_param);

            // Apply Convert if original had one
            ov::Output<ov::Node> param_for_add;
            if (convert_node) {
                auto new_convert =
                    std::make_shared<ov::opset1::Convert>(new_param, convert_node->get_destination_type());
                new_convert->set_friendly_name(convert_node->get_friendly_name() + "/branch_" + std::to_string(i));
                param_for_add = new_convert->output(0);
            } else {
                param_for_add = new_param->output(0);
            }

            // Create Add or Multiply: concat_input[i] op param[i]
            std::shared_ptr<ov::Node> new_elementwise;
            if (add_op) {
                new_elementwise = std::make_shared<ov::opset1::Add>(concat_inputs[i], param_for_add);
            } else {
                new_elementwise = std::make_shared<ov::opset1::Multiply>(concat_inputs[i], param_for_add);
            }
            new_elementwise->set_friendly_name(elementwise_node->get_friendly_name() + "/branch_" + std::to_string(i));
            new_add_ops.push_back(new_elementwise);
        }

        auto new_concat = std::make_shared<ov::opset1::Concat>(new_add_ops, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name() + "/after_" + op_type);

        // Register new parameters with model
        model_->add_parameters(new_params);

        ov::NodeVector from_nodes = {elementwise_node, concat};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(elementwise_node, new_concat);

        LOG_INFO("  Pushed " << op_type << " into " << num_branches << " branches with new parameters");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(elementwise_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushSliceBeforeConcat
// =============================================================================
// Transforms: Concat([a,b,c], axis=0) → Slice(axis=k)  =>  Concat([Slice(a,axis=k), Slice(b,axis=k), Slice(c,axis=k)])
// Safety constraint: Only safe when slice axis != concat axis
// Rationale: Slicing on concat axis would produce incorrect results after transformation

PushSliceBeforeConcat::PushSliceBeforeConcat() {
    MATCHER_SCOPE(PushSliceBeforeConcat);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto slice_pattern = ov::pass::pattern::wrap_type<ov::op::v8::Slice>({concat_pattern,
                                                                          ov::pass::pattern::any_input(),
                                                                          ov::pass::pattern::any_input(),
                                                                          ov::pass::pattern::any_input(),
                                                                          ov::pass::pattern::any_input()});

    auto callback = [concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(m.get_match_root());
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!slice || !concat)
            return false;

        // Get slice parameters: start, stop, step, axes
        auto start_const = std::dynamic_pointer_cast<ov::opset1::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto stop_const = std::dynamic_pointer_cast<ov::opset1::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto step_const = std::dynamic_pointer_cast<ov::opset1::Constant>(slice->input_value(3).get_node_shared_ptr());
        auto axes_const = std::dynamic_pointer_cast<ov::opset1::Constant>(slice->input_value(4).get_node_shared_ptr());

        if (!start_const || !stop_const || !step_const || !axes_const) {
            LOG_DEBUG("  Slice parameters are not constants, skipping");
            return false;
        }

        auto axes_data = axes_const->cast_vector<int64_t>();
        if (axes_data.empty())
            return false;

        int64_t concat_axis = concat->get_axis();

        // Check if any slice axis matches concat axis
        for (auto axis : axes_data) {
            if (axis == concat_axis) {
                LOG_DEBUG("  Slice axis matches Concat axis, not safe to push, skipping");
                return false;
            }
        }

        LOG_INFO("PushSliceBeforeConcat: Pushing Slice before Concat");
        LOG_INFO("  Concat axis: " << concat_axis << ", Slice axes: " << axes_data[0]);

        auto concat_inputs = concat->input_values();
        ov::NodeVector new_sliced_outputs;

        for (size_t i = 0; i < concat_inputs.size(); ++i) {
            auto new_slice = std::make_shared<ov::op::v8::Slice>(concat_inputs[i],
                                                                 slice->input_value(1),   // start
                                                                 slice->input_value(2),   // stop
                                                                 slice->input_value(3),   // step
                                                                 slice->input_value(4));  // axes
            new_slice->set_friendly_name(slice->get_friendly_name() + "/branch_" + std::to_string(i));
            new_sliced_outputs.push_back(new_slice);
        }

        auto new_concat = std::make_shared<ov::opset1::Concat>(new_sliced_outputs, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name() + "/after_Slice");

        ov::NodeVector from_nodes = {slice, concat};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(slice, new_concat);

        LOG_INFO("  Pushed Slice into " << concat_inputs.size() << " branches");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(slice_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushClampBeforeConcat
// =============================================================================
// Transforms: Concat([a,b,c]) → Clamp  =>  Concat([Clamp(a), Clamp(b), Clamp(c)])

PushClampBeforeConcat::PushClampBeforeConcat() {
    MATCHER_SCOPE(PushClampBeforeConcat);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto clamp_pattern = ov::pass::pattern::wrap_type<ov::opset1::Clamp>({concat_pattern});

    auto callback = [concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto clamp = std::dynamic_pointer_cast<ov::opset1::Clamp>(m.get_match_root());
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!clamp || !concat)
            return false;

        LOG_INFO("PushClampBeforeConcat: Pushing Clamp before Concat");
        LOG_INFO("  Clamp min: " << clamp->get_min() << ", max: " << clamp->get_max());

        auto concat_inputs = concat->input_values();
        ov::NodeVector new_clamped_outputs;

        for (size_t i = 0; i < concat_inputs.size(); ++i) {
            auto new_clamp = std::make_shared<ov::opset1::Clamp>(concat_inputs[i], clamp->get_min(), clamp->get_max());
            new_clamp->set_friendly_name(clamp->get_friendly_name() + "/branch_" + std::to_string(i));
            new_clamped_outputs.push_back(new_clamp);
        }

        auto new_concat = std::make_shared<ov::opset1::Concat>(new_clamped_outputs, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name() + "/after_Clamp");

        ov::NodeVector from_nodes = {clamp, concat};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(clamp, new_concat);

        LOG_INFO("  Pushed Clamp into " << concat_inputs.size() << " branches");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(clamp_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushScalarElementwiseBeforeConcat
// =============================================================================
// Transforms: Concat([a,b,c]) → Add/Minimum/Swish(scalar)  =>  Concat([Op(a,scalar), Op(b,scalar), Op(c,scalar)])
// Constraint: scalar input must have exactly 1 element (shape like [1,1,1])
// Optimization: Broadcasting scalar operation can be distributed into branches

PushScalarElementwiseBeforeConcat::PushScalarElementwiseBeforeConcat() {
    MATCHER_SCOPE(PushScalarElementwiseBeforeConcat);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto elementwise_pattern = ov::pass::pattern::wrap_type<ov::opset1::Add, ov::opset1::Minimum, ov::op::v4::Swish>(
        {concat_pattern, ov::pass::pattern::any_input()});

    auto callback = [concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto elementwise = m.get_match_root();
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!elementwise || !concat)
            return false;

        // Find which input is concat and which is the scalar
        ov::Output<ov::Node> scalar_input;
        size_t concat_input_idx = 0;

        if (elementwise->input_value(0).get_node_shared_ptr() == concat) {
            scalar_input = elementwise->input_value(1);
            concat_input_idx = 0;
        } else if (elementwise->input_value(1).get_node_shared_ptr() == concat) {
            scalar_input = elementwise->input_value(0);
            concat_input_idx = 1;
        } else {
            return false;
        }

        // Check if scalar_input has shape like 1x1x1 (single element)
        auto scalar_shape = scalar_input.get_partial_shape();
        if (!scalar_shape.rank().is_static()) {
            return false;
        }

        size_t total_elements = 1;
        for (int64_t i = 0; i < scalar_shape.rank().get_length(); ++i) {
            if (!scalar_shape[i].is_static()) {
                return false;
            }
            total_elements *= scalar_shape[i].get_length();
        }

        if (total_elements != 1) {
            LOG_DEBUG("  Scalar input has " << total_elements << " elements, not a scalar, skipping");
            return false;
        }

        std::string op_type = elementwise->get_type_name();
        LOG_INFO("PushScalarElementwiseBeforeConcat: Pushing " << op_type << " with scalar before Concat");
        LOG_INFO("  Scalar shape: " << scalar_shape);

        auto concat_inputs = concat->input_values();
        ov::NodeVector new_elementwise_outputs;

        for (size_t i = 0; i < concat_inputs.size(); ++i) {
            std::shared_ptr<ov::Node> new_elementwise;

            if (auto add_op = std::dynamic_pointer_cast<ov::opset1::Add>(elementwise)) {
                if (concat_input_idx == 0) {
                    new_elementwise = std::make_shared<ov::opset1::Add>(concat_inputs[i], scalar_input);
                } else {
                    new_elementwise = std::make_shared<ov::opset1::Add>(scalar_input, concat_inputs[i]);
                }
            } else if (auto min_op = std::dynamic_pointer_cast<ov::opset1::Minimum>(elementwise)) {
                if (concat_input_idx == 0) {
                    new_elementwise = std::make_shared<ov::opset1::Minimum>(concat_inputs[i], scalar_input);
                } else {
                    new_elementwise = std::make_shared<ov::opset1::Minimum>(scalar_input, concat_inputs[i]);
                }
            } else if (auto swish_op = std::dynamic_pointer_cast<ov::op::v4::Swish>(elementwise)) {
                // Swish has input order: data, beta (scalar)
                if (concat_input_idx == 0) {
                    new_elementwise = std::make_shared<ov::op::v4::Swish>(concat_inputs[i], scalar_input);
                } else {
                    new_elementwise = std::make_shared<ov::op::v4::Swish>(scalar_input, concat_inputs[i]);
                }
            } else {
                return false;
            }

            new_elementwise->set_friendly_name(elementwise->get_friendly_name() + "/branch_" + std::to_string(i));
            new_elementwise_outputs.push_back(new_elementwise);
        }

        auto new_concat = std::make_shared<ov::opset1::Concat>(new_elementwise_outputs, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name() + "/after_" + op_type);

        ov::NodeVector from_nodes = {elementwise, concat};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(elementwise, new_concat);

        LOG_INFO("  Pushed " << op_type << " into " << concat_inputs.size() << " branches");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(elementwise_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushMultiplyBeforeConcat
// =============================================================================
// Transforms: Concat([a,b,c]) * Concat([d,e,f])  =>  Concat([a*d, b*e, c*f])
// Requirements: Both Concat operations must have:
//   1. Same concatenation axis
//   2. Same number of inputs
//   3. Pairwise compatible input shapes
// Optimization: Fuses two concat operations by distributing Multiply element-wise

PushMultiplyBeforeConcat::PushMultiplyBeforeConcat() {
    MATCHER_SCOPE(PushMultiplyBeforeConcat);

    auto concat1_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto concat2_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto multiply_pattern = ov::pass::pattern::wrap_type<ov::opset1::Multiply>({concat1_pattern, concat2_pattern});

    auto callback = [concat1_pattern, concat2_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto multiply = std::dynamic_pointer_cast<ov::opset1::Multiply>(m.get_match_root());
        auto concat1 = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat1_pattern).get_node_shared_ptr());
        auto concat2 = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat2_pattern).get_node_shared_ptr());

        if (!multiply || !concat1 || !concat2)
            return false;

        // Check if both Concat have the same axis
        if (concat1->get_axis() != concat2->get_axis()) {
            LOG_DEBUG("  Concat axes differ, skipping");
            return false;
        }

        auto concat1_inputs = concat1->input_values();
        auto concat2_inputs = concat2->input_values();

        // Check if both Concat have the same number of inputs
        if (concat1_inputs.size() != concat2_inputs.size()) {
            LOG_DEBUG("  Concat input counts differ, skipping");
            return false;
        }

        // Check if all corresponding inputs have matching shapes
        bool shapes_match = true;
        for (size_t i = 0; i < concat1_inputs.size(); ++i) {
            auto shape1 = concat1_inputs[i].get_partial_shape();
            auto shape2 = concat2_inputs[i].get_partial_shape();

            if (!shape1.compatible(shape2)) {
                shapes_match = false;
                break;
            }
        }

        if (!shapes_match) {
            LOG_DEBUG("  Concat input shapes are not compatible, skipping");
            return false;
        }

        LOG_INFO("PushMultiplyBeforeConcat: Pushing Multiply before Concat with matching input shapes");
        LOG_INFO("  Moving Multiply before Concat with " << concat1_inputs.size() << " branches");

        ov::NodeVector new_multiply_outputs;

        for (size_t i = 0; i < concat1_inputs.size(); ++i) {
            auto new_multiply = std::make_shared<ov::opset1::Multiply>(concat1_inputs[i], concat2_inputs[i]);
            new_multiply->set_friendly_name(multiply->get_friendly_name() + "/branch_" + std::to_string(i));
            new_multiply_outputs.push_back(new_multiply);
        }

        auto new_concat = std::make_shared<ov::opset1::Concat>(new_multiply_outputs, concat1->get_axis());
        new_concat->set_friendly_name(concat1->get_friendly_name() + "/after_Multiply");

        ov::NodeVector from_nodes = {multiply, concat1, concat2};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(multiply, new_concat);

        LOG_INFO("  Created " << concat1_inputs.size() << " Multiply operations before single Concat");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// UnrollConcatMatMul
// =============================================================================
// Transforms MatMul with Concat input and parameter-based weights to N expert branches
// Pattern: Concat([a,b,c,d]) ────────┐
//          scale_param + weights_param → multiply → convert ─┤→ MatMul
//
// Input requirements:
//   - Concat inputs: each branch has shape like [1,1,1,H]
//   - scale_param shape: [N,A,1] where N = number of experts
//   - weights_param shape: [N,A,B]
// Output: N expert branches, each with new parameters [1,A,1] and [1,A,B]
// Optimization: Unrolls batched expert computation for parallel execution

UnrollConcatMatMul::UnrollConcatMatMul(std::shared_ptr<ov::Model> model) : model_(model) {
    MATCHER_SCOPE(UnrollConcatMatMul);

    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::opset1::MatMul>();

    auto callback = [this](ov::pass::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ov::opset1::MatMul>(m.get_match_root());
        if (!matmul)
            return false;

        LOG_INFO("UnrollConcatMatMul: Checking MatMul " << matmul->get_friendly_name());

        auto matmul_input0 = matmul->input_value(0);  // Should be Concat
        auto matmul_input1 = matmul->input_value(1);  // Should be Multiply → Convert chain

        // Check input0: should be Concat
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(matmul_input0.get_node_shared_ptr());
        if (!concat) {
            LOG_DEBUG("  Input0 is not Concat, skipping");
            return false;
        }

        size_t num_branches = concat->get_input_size();
        LOG_DEBUG("  Found Concat with " << num_branches << " branches");

        // Check input1: Multiply (possibly through Convert)
        auto input1_node = matmul_input1.get_node_shared_ptr();
        std::shared_ptr<ov::opset1::Convert> convert_after_multiply;
        std::shared_ptr<ov::opset1::Multiply> multiply_node;

        if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(input1_node)) {
            convert_after_multiply = conv;
            multiply_node = std::dynamic_pointer_cast<ov::opset1::Multiply>(conv->input_value(0).get_node_shared_ptr());
        } else {
            multiply_node = std::dynamic_pointer_cast<ov::opset1::Multiply>(input1_node);
        }

        if (!multiply_node) {
            LOG_DEBUG("  Input1 is not Multiply, skipping");
            return false;
        }

        auto multiply_input0 = multiply_node->input_value(0);
        auto multiply_input1 = multiply_node->input_value(1);

        // Helper: skip Convert to get to Parameter
        auto skip_convert = [](ov::Output<ov::Node> out) -> ov::Output<ov::Node> {
            if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(out.get_node_shared_ptr())) {
                return conv->input_value(0);
            }
            return out;
        };

        // Helper: calculate total size
        auto calc_total_size = [](const ov::PartialShape& shape) -> size_t {
            if (!shape.rank().is_static())
                return 0;
            size_t total = 1;
            for (int64_t i = 0; i < shape.rank().get_length(); ++i) {
                if (shape[i].is_static()) {
                    total *= shape[i].get_length();
                }
            }
            return total;
        };

        auto mult_in0_skip_convert = skip_convert(multiply_input0);
        auto mult_in1_skip_convert = skip_convert(multiply_input1);

        size_t size0 = calc_total_size(mult_in0_skip_convert.get_partial_shape());
        size_t size1 = calc_total_size(mult_in1_skip_convert.get_partial_shape());

        // Determine scale vs weights by total size (larger = weights)
        ov::Output<ov::Node> scale_param_source, weights_param_source;
        if (size0 > size1) {
            weights_param_source = multiply_input0;
            scale_param_source = multiply_input1;
        } else {
            weights_param_source = multiply_input1;
            scale_param_source = multiply_input0;
        }

        // Extract parameter nodes
        auto get_param_node = [](ov::Output<ov::Node> out) -> std::shared_ptr<ov::op::v0::Parameter> {
            auto node = out.get_node_shared_ptr();
            if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(node)) {
                node = conv->input_value(0).get_node_shared_ptr();
            }
            return std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        };

        auto scale_param = get_param_node(scale_param_source);
        auto weights_param = get_param_node(weights_param_source);

        if (!scale_param || !weights_param) {
            LOG_ERROR("  Could not find scale or weights parameter nodes");
            return false;
        }

        // Get original shapes: [N, A, 1] and [N, A, B]
        auto scale_orig_shape = scale_param->get_partial_shape();
        auto weights_orig_shape = weights_param->get_partial_shape();

        if (!scale_orig_shape.rank().is_static() || !weights_orig_shape.rank().is_static()) {
            LOG_DEBUG("  Parameter shapes are not static, skipping");
            return false;
        }

        if (!scale_orig_shape[0].is_static() || !weights_orig_shape[0].is_static()) {
            LOG_DEBUG("  First dimension is not static, skipping");
            return false;
        }

        size_t scale_num_experts = scale_orig_shape[0].get_length();
        size_t weights_num_experts = weights_orig_shape[0].get_length();

        if (scale_num_experts != weights_num_experts || scale_num_experts != num_branches) {
            LOG_DEBUG("  Number of experts mismatch: scale=" << scale_num_experts << ", weights=" << weights_num_experts
                                                             << ", concat_branches=" << num_branches);
            return false;
        }

        LOG_INFO("  Found pattern: Concat + scale_param + weights_param → Multiply → MatMul");
        LOG_INFO("  Creating " << num_branches << " expert branches...");

        // Create per-expert parameter shapes: [N,...] → [1,...]
        auto scale_shape_vec = scale_orig_shape.to_shape();
        auto weights_shape_vec = weights_orig_shape.to_shape();

        ov::Shape scale_new_shape = scale_shape_vec;
        ov::Shape weights_new_shape = weights_shape_vec;
        scale_new_shape[0] = 1;
        weights_new_shape[0] = 1;

        LOG_INFO("  Scale: " << scale_shape_vec << " → " << scale_new_shape);
        LOG_INFO("  Weights: " << weights_shape_vec << " → " << weights_new_shape);

        auto concat_inputs = concat->input_values();
        ov::NodeVector expert_outputs;
        ov::ParameterVector new_params;

        for (size_t expert_idx = 0; expert_idx < num_branches; ++expert_idx) {
            // 1. Get Concat input for this branch
            auto concat_input = concat_inputs[expert_idx];

            // 2. Create new scale parameter
            auto new_scale_param = std::make_shared<ov::op::v0::Parameter>(scale_param->get_element_type(),
                                                                           ov::PartialShape(scale_new_shape));
            new_scale_param->set_friendly_name(scale_param->get_friendly_name() + "/expert_" +
                                               std::to_string(expert_idx));
            new_params.push_back(new_scale_param);

            // 3. Create new weights parameter
            auto new_weights_param = std::make_shared<ov::op::v0::Parameter>(weights_param->get_element_type(),
                                                                             ov::PartialShape(weights_new_shape));
            new_weights_param->set_friendly_name(weights_param->get_friendly_name() + "/expert_" +
                                                 std::to_string(expert_idx));
            new_params.push_back(new_weights_param);

            // 4. Apply Convert to weights if needed
            ov::Output<ov::Node> weights_for_multiply;
            if (auto weights_convert =
                    std::dynamic_pointer_cast<ov::opset1::Convert>(weights_param_source.get_node_shared_ptr())) {
                auto new_weights_convert =
                    std::make_shared<ov::opset1::Convert>(new_weights_param, weights_convert->get_destination_type());
                new_weights_convert->set_friendly_name(weights_convert->get_friendly_name() + "/expert_" +
                                                       std::to_string(expert_idx));
                weights_for_multiply = new_weights_convert->output(0);
            } else {
                weights_for_multiply = new_weights_param->output(0);
            }

            // 5. Apply Convert to scale if needed
            ov::Output<ov::Node> scale_for_multiply;
            if (auto scale_convert =
                    std::dynamic_pointer_cast<ov::opset1::Convert>(scale_param_source.get_node_shared_ptr())) {
                auto new_scale_convert =
                    std::make_shared<ov::opset1::Convert>(new_scale_param, scale_convert->get_destination_type());
                new_scale_convert->set_friendly_name(scale_convert->get_friendly_name() + "/expert_" +
                                                     std::to_string(expert_idx));
                scale_for_multiply = new_scale_convert->output(0);
            } else {
                scale_for_multiply = new_scale_param->output(0);
            }

            // 6. Multiply: scale * weights
            auto new_multiply = std::make_shared<ov::opset1::Multiply>(scale_for_multiply, weights_for_multiply);
            new_multiply->set_friendly_name(multiply_node->get_friendly_name() + "/expert_" +
                                            std::to_string(expert_idx));

            // 7. Convert after Multiply if needed
            ov::Output<ov::Node> weights_for_matmul;
            if (convert_after_multiply) {
                auto new_convert_after_multiply =
                    std::make_shared<ov::opset1::Convert>(new_multiply, convert_after_multiply->get_destination_type());
                new_convert_after_multiply->set_friendly_name(convert_after_multiply->get_friendly_name() + "/expert_" +
                                                              std::to_string(expert_idx));
                weights_for_matmul = new_convert_after_multiply->output(0);
            } else {
                weights_for_matmul = new_multiply->output(0);
            }

            // 8. MatMul: concat_input × weights
            auto new_matmul = std::make_shared<ov::opset1::MatMul>(concat_input,
                                                                   weights_for_matmul,
                                                                   matmul->get_transpose_a(),
                                                                   matmul->get_transpose_b());
            new_matmul->set_friendly_name(matmul->get_friendly_name() + "/expert_" + std::to_string(expert_idx));

            expert_outputs.push_back(new_matmul);
        }

        // Concat all expert outputs
        auto output_concat = std::make_shared<ov::opset1::Concat>(expert_outputs, concat->get_axis());
        output_concat->set_friendly_name(matmul->get_friendly_name() + "/concat");

        // Register new parameters with model
        model_->add_parameters(new_params);

        // Replace original MatMul
        ov::copy_runtime_info(matmul, output_concat);
        ov::replace_node(matmul, output_concat);

        LOG_INFO("  Successfully created " << num_branches << " expert branches with " << new_params.size()
                                           << " new parameters");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// PushReshapeBeforeConcat
// =============================================================================
// Transforms: Concat([a,b,c,d], axis=0) → Reshape  =>  Concat([Reshape(a), Reshape(b), Reshape(c), Reshape(d)])
// Example: Concat(4x1x2880) → Reshape(4x1x1x2880) becomes Concat([1x1x2880→1x1x1x2880, ...])
//
// Safety constraints:
//   1. Concat axis must be 0 (highest dimension)
//   2. Reshape must preserve dimension 0 (e.g., [N,...] → [N,...,extra])
// Rationale: Ensures reshape can be safely distributed to each branch

PushReshapeBeforeConcat::PushReshapeBeforeConcat() {
    MATCHER_SCOPE(PushReshapeBeforeConcat);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto reshape_pattern =
        ov::pass::pattern::wrap_type<ov::opset1::Reshape>({concat_pattern, ov::pass::pattern::any_input()});

    auto callback = [concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto reshape = std::dynamic_pointer_cast<ov::opset1::Reshape>(m.get_match_root());
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!reshape || !concat)
            return false;

        // Get reshape target shape
        auto reshape_pattern_const =
            std::dynamic_pointer_cast<ov::opset1::Constant>(reshape->input_value(1).get_node_shared_ptr());

        if (!reshape_pattern_const) {
            LOG_DEBUG("  Reshape pattern is not a Constant, skipping");
            return false;
        }

        auto concat_output_shape = concat->get_output_partial_shape(0);
        auto reshape_output_shape = reshape->get_output_partial_shape(0);

        if (!concat_output_shape.rank().is_static() || !reshape_output_shape.rank().is_static()) {
            LOG_DEBUG("  Shapes are not static, skipping");
            return false;
        }

        int64_t concat_axis = concat->get_axis();

        // Strict check: concat axis must be 0 (highest dimension)
        if (concat_axis != 0) {
            LOG_DEBUG("  Concat axis is not 0, skipping");
            return false;
        }

        auto concat_inputs = concat->input_values();
        size_t num_branches = concat_inputs.size();

        // Check that reshape output shape[0] == concat output shape[0]
        if (!concat_output_shape[0].is_static() || !reshape_output_shape[0].is_static()) {
            LOG_DEBUG("  First dimension is not static, skipping");
            return false;
        }

        int64_t concat_dim0 = concat_output_shape[0].get_length();
        int64_t reshape_dim0 = reshape_output_shape[0].get_length();

        if (concat_dim0 != reshape_dim0) {
            LOG_DEBUG("  Concat shape[0]=" << concat_dim0 << " != Reshape shape[0]=" << reshape_dim0 << ", skipping");
            return false;
        }

        LOG_INFO("PushReshapeBeforeConcat: Pushing Reshape before Concat");
        LOG_INFO("  Concat shape: " << concat_output_shape << " → Reshape shape: " << reshape_output_shape);
        LOG_INFO("  Concat axis: " << concat_axis << ", num branches: " << num_branches);

        // Calculate per-branch reshape target shape
        // Concat output: [N, ...], Reshape output: [N, ..., extra_dims]
        // Branch input: [1, ...], Branch reshape: [1, ..., extra_dims]
        auto reshape_shape_vec = reshape_pattern_const->cast_vector<int64_t>();
        std::vector<int64_t> branch_reshape_shape = reshape_shape_vec;

        // Set first dimension to 1 (per branch)
        branch_reshape_shape[0] = 1;

        ov::NodeVector new_reshaped_outputs;

        for (size_t i = 0; i < num_branches; ++i) {
            auto new_reshape_shape_const =
                std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                       ov::Shape{branch_reshape_shape.size()},
                                                       branch_reshape_shape);

            auto new_reshape = std::make_shared<ov::opset1::Reshape>(concat_inputs[i],
                                                                     new_reshape_shape_const,
                                                                     reshape->get_special_zero());
            new_reshape->set_friendly_name(reshape->get_friendly_name() + "/branch_" + std::to_string(i));
            new_reshaped_outputs.push_back(new_reshape);
        }

        // New concat also on axis 0 (highest dimension)
        auto new_concat = std::make_shared<ov::opset1::Concat>(new_reshaped_outputs, 0);
        new_concat->set_friendly_name(concat->get_friendly_name() + "/after_Reshape");

        ov::NodeVector from_nodes = {reshape, concat};
        ov::copy_runtime_info(from_nodes, new_concat);
        ov::replace_node(reshape, new_concat);

        LOG_INFO("  Pushed Reshape into " << num_branches << " branches");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// FuseConcatReduceSum
// =============================================================================
// Transforms: Concat([a,b,c], axis=0) → ReduceSum(axis=0)  =>  a + b + c
// Optimizes sum reduction over concatenation axis by converting to cascaded Add
// Handles both keep_dims=true and keep_dims=false cases

FuseConcatReduceSum::FuseConcatReduceSum() {
    MATCHER_SCOPE(FuseConcatReduceSum);

    auto concat_pattern = ov::pass::pattern::wrap_type<ov::opset1::Concat>();
    auto reduce_sum_pattern =
        ov::pass::pattern::wrap_type<ov::opset1::ReduceSum>({concat_pattern, ov::pass::pattern::any_input()});

    auto callback = [concat_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto reduce_sum = std::dynamic_pointer_cast<ov::opset1::ReduceSum>(m.get_match_root());
        auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(pm.at(concat_pattern).get_node_shared_ptr());

        if (!reduce_sum || !concat)
            return false;

        // Validate: reduction axes must be constant
        auto reduce_axes_const =
            std::dynamic_pointer_cast<ov::opset1::Constant>(reduce_sum->input_value(1).get_node_shared_ptr());
        if (!reduce_axes_const)
            return false;

        // Validate: only single axis reduction matching concat axis is safe to optimize
        // If axes differ, the transformation would produce incorrect results
        auto reduce_axes = reduce_axes_const->cast_vector<int64_t>();
        if (reduce_axes.size() != 1 || reduce_axes[0] != concat->get_axis()) {
            return false;
        }

        // Check keep_dims attribute to determine output shape handling
        // keep_dims=false: ReduceSum([N,H], axis=0) → [H] (removes axis)
        // keep_dims=true:  ReduceSum([N,H], axis=0) → [1,H] (keeps axis with size 1)
        bool keep_dims = reduce_sum->get_keep_dims();

        LOG_INFO("FuseConcatReduceSum: Fusing Concat+ReduceSum to cascaded Add");
        LOG_INFO("  keep_dims: " << keep_dims);

        auto inputs = concat->input_values();
        if (inputs.size() < 2)
            return false;

        // Prepare inputs based on keep_dims to match output shape
        // Example: Concat([a,b,c], axis=0) → ReduceSum(axis=0)
        //   If keep_dims=false: need Squeeze([1,H])→[H] for each input, then a+b+c → [H]
        //   If keep_dims=true:  directly add [1,H] inputs, then a+b+c → [1,H]
        ov::OutputVector prepared_inputs;
        if (!keep_dims) {
            // ReduceSum removes the reduction axis, so Squeeze concat inputs to match
            auto squeeze_axes = std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                                       ov::Shape{1},
                                                                       std::vector<int64_t>{concat->get_axis()});
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto squeeze = std::make_shared<ov::opset1::Squeeze>(inputs[i], squeeze_axes);
                squeeze->set_friendly_name(reduce_sum->get_friendly_name() + "/squeeze_" + std::to_string(i));
                prepared_inputs.push_back(squeeze->output(0));
            }
        } else {
            // keep_dims=true: ReduceSum preserves the axis dimension, Add directly
            for (size_t i = 0; i < inputs.size(); ++i) {
                prepared_inputs.push_back(inputs[i]);
            }
        }

        // Create cascaded Add operations
        ov::Output<ov::Node> accumulated = prepared_inputs[0];
        for (size_t i = 1; i < prepared_inputs.size(); ++i) {
            auto add_node = std::make_shared<ov::opset1::Add>(accumulated, prepared_inputs[i]);
            add_node->set_friendly_name(reduce_sum->get_friendly_name() + "/add_" + std::to_string(i));
            accumulated = add_node->output(0);
        }

        ov::NodeVector from_nodes = {reduce_sum, concat};
        ov::copy_runtime_info(from_nodes, accumulated.get_node_shared_ptr());
        ov::replace_node(reduce_sum, accumulated.get_node_shared_ptr());

        LOG_INFO("  Created " << (inputs.size() - 1) << " Add operations");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_sum_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// UnrollExpertReshape
// =============================================================================
// Transforms: Tile → Reshape[num_experts,1,hidden]  =>  N×Reshape[1,1,hidden] → Concat
// Purpose: Unrolls expert dimension in Reshape to enable per-expert optimization
// Requirements:
//   - Reshape output must be 3D with shape [num_experts, 1, hidden_dim]
//   - First dimension must equal num_experts
// Output: Creates N separate Reshape operations, one per expert, concatenated on axis 0

UnrollExpertReshape::UnrollExpertReshape(size_t num_experts) : num_experts_(num_experts) {
    MATCHER_SCOPE(UnrollExpertReshape);

    auto tile_pattern = ov::pass::pattern::wrap_type<ov::opset1::Tile>();
    auto reshape_pattern =
        ov::pass::pattern::wrap_type<ov::opset1::Reshape>({tile_pattern, ov::pass::pattern::any_input()});

    auto callback = [this, tile_pattern](ov::pass::pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();
        auto reshape = std::dynamic_pointer_cast<ov::opset1::Reshape>(m.get_match_root());
        auto tile = std::dynamic_pointer_cast<ov::opset1::Tile>(pm.at(tile_pattern).get_node_shared_ptr());

        if (!reshape || !tile)
            return false;

        auto reshape_output_shape = reshape->get_output_partial_shape(0);
        if (!reshape_output_shape.rank().is_static() || reshape_output_shape.rank().get_length() != 3 ||
            !reshape_output_shape[0].is_static() ||
            reshape_output_shape[0].get_length() != static_cast<int64_t>(num_experts_)) {
            return false;
        }

        LOG_INFO("UnrollExpertReshape: Unrolling Reshape after Tile");

        auto tile_input = tile->input_value(0);
        ov::NodeVector expert_reshapes;

        for (size_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
            auto new_shape_const = std::make_shared<ov::opset1::Constant>(
                ov::element::i64,
                ov::Shape{3},
                std::vector<int64_t>{1, 1, static_cast<int64_t>(reshape_output_shape[2].get_length())});

            auto expert_reshape = std::make_shared<ov::opset1::Reshape>(tile_input, new_shape_const, false);
            expert_reshape->set_friendly_name(reshape->get_friendly_name() + "/expert_" + std::to_string(expert_idx));
            expert_reshapes.push_back(expert_reshape);
        }

        auto concat = std::make_shared<ov::opset1::Concat>(expert_reshapes, 0);
        concat->set_friendly_name(reshape->get_friendly_name() + "/concat");

        ov::NodeVector from_nodes = {tile, reshape};
        ov::copy_runtime_info(from_nodes, concat);
        ov::replace_node(reshape, concat);

        LOG_INFO("  Created " << num_experts_ << " expert Reshape branches");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_pattern, matcher_name);
    register_matcher(m, callback);
}

// =============================================================================
// RemoveUnusedParameters: Clean up parameters that have no consumers
// =============================================================================

RemoveUnusedParameters::RemoveUnusedParameters(std::shared_ptr<ov::Model> model) : model_(model) {
    MATCHER_SCOPE(RemoveUnusedParameters);

    // Match any Result node to trigger cleanup after all other patterns
    auto result_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Result>();

    auto callback = [this](ov::pass::pattern::Matcher& m) {
        LOG_INFO("RemoveUnusedParameters: Scanning for unused parameters");

        auto params = model_->get_parameters();
        ov::ParameterVector unused_params;

        for (const auto& param : params) {
            bool has_consumers = false;

            // Check if parameter output has any target inputs
            for (const auto& output : param->outputs()) {
                if (!output.get_target_inputs().empty()) {
                    has_consumers = true;
                    break;
                }
            }

            if (!has_consumers) {
                unused_params.push_back(param);
                LOG_INFO("  Found unused parameter: " << param->get_friendly_name());
            }
        }

        if (!unused_params.empty()) {
            LOG_INFO("  Removing " << unused_params.size() << " unused parameters");
            for (const auto& param : unused_params) {
                model_->remove_parameter(param);
            }
            return true;  // Model was modified
        } else {
            LOG_INFO("  No unused parameters found");
            return false;  // Model was not modified
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
