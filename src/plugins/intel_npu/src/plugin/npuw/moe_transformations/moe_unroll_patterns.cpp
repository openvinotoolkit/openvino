// Copyright (C) 2018-2026 Intel Corporation
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

namespace {

// =============================================================================
// Common Helper Functions
// =============================================================================

/**
 * @brief Skip Convert node if present and return the underlying input
 */
inline ov::Output<ov::Node> skip_convert(ov::Output<ov::Node> out) {
    if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(out.get_node_shared_ptr())) {
        return conv->input_value(0);
    }
    return out;
}

/**
 * @brief Calculate total number of elements in a shape
 */
inline size_t calc_total_size(const ov::PartialShape& shape) {
    if (!shape.rank().is_static())
        return 0;
    size_t total = 1;
    for (int64_t i = 0; i < shape.rank().get_length(); ++i) {
        if (shape[i].is_static()) {
            total *= shape[i].get_length();
        }
    }
    return total;
}

/**
 * @brief Extract Parameter node from output, skipping intermediate Convert if present
 */
inline std::shared_ptr<ov::op::v0::Parameter> get_param_node(ov::Output<ov::Node> out) {
    auto skipped = skip_convert(out);
    return std::dynamic_pointer_cast<ov::op::v0::Parameter>(skipped.get_node_shared_ptr());
}

/**
 * @brief Context for creating expert branch with scale and weights parameters
 */
struct ExpertBranchContext {
    size_t expert_idx;
    std::shared_ptr<ov::op::v0::Parameter> scale_param;
    std::shared_ptr<ov::op::v0::Parameter> weights_param;
    ov::Output<ov::Node> scale_param_source;
    ov::Output<ov::Node> weights_param_source;
    std::shared_ptr<ov::opset1::Multiply> multiply_node;
    std::shared_ptr<ov::opset1::Convert> convert_after_multiply;
    std::shared_ptr<ov::opset1::MatMul> matmul;
    ov::Shape scale_new_shape;
    ov::Shape weights_new_shape;
};

/**
 * @brief Create expert branch: parameters → convert → multiply → convert → weights output
 *
 * This helper function encapsulates the common logic for creating per-expert parameters
 * and weight computation chain used in UnrollMoEMatMul for all input patterns.
 *
 * @param ctx Context containing all necessary information for creating the branch
 * @param new_params Output vector to collect newly created parameters
 * @return Processed weights ready for MatMul operation
 */
inline ov::Output<ov::Node> create_expert_branch_weights(const ExpertBranchContext& ctx,
                                                         ov::ParameterVector& new_params) {
    // 1. Create new scale parameter
    auto new_scale_param = std::make_shared<ov::op::v0::Parameter>(ctx.scale_param->get_element_type(),
                                                                   ov::PartialShape(ctx.scale_new_shape));
    new_scale_param->set_friendly_name(ctx.scale_param->get_friendly_name() + "/expert_" +
                                       std::to_string(ctx.expert_idx));
    new_scale_param->get_rt_info()["moe_original_param"] = ctx.scale_param->get_friendly_name();
    new_scale_param->get_rt_info()["moe_expert_index"] = static_cast<int64_t>(ctx.expert_idx);
    new_params.push_back(new_scale_param);

    // 2. Create new weights parameter
    auto new_weights_param = std::make_shared<ov::op::v0::Parameter>(ctx.weights_param->get_element_type(),
                                                                     ov::PartialShape(ctx.weights_new_shape));
    new_weights_param->set_friendly_name(ctx.weights_param->get_friendly_name() + "/expert_" +
                                         std::to_string(ctx.expert_idx));
    new_weights_param->get_rt_info()["moe_original_param"] = ctx.weights_param->get_friendly_name();
    new_weights_param->get_rt_info()["moe_expert_index"] = static_cast<int64_t>(ctx.expert_idx);
    new_params.push_back(new_weights_param);

    // 3. Apply Convert to weights if needed
    ov::Output<ov::Node> weights_for_multiply;
    if (auto weights_convert =
            std::dynamic_pointer_cast<ov::opset1::Convert>(ctx.weights_param_source.get_node_shared_ptr())) {
        auto new_weights_convert =
            std::make_shared<ov::opset1::Convert>(new_weights_param, weights_convert->get_destination_type());
        new_weights_convert->set_friendly_name(weights_convert->get_friendly_name() + "/expert_" +
                                               std::to_string(ctx.expert_idx));
        weights_for_multiply = new_weights_convert->output(0);
    } else {
        weights_for_multiply = new_weights_param->output(0);
    }

    // 4. Apply Convert to scale if needed
    ov::Output<ov::Node> scale_for_multiply;
    if (auto scale_convert =
            std::dynamic_pointer_cast<ov::opset1::Convert>(ctx.scale_param_source.get_node_shared_ptr())) {
        auto new_scale_convert =
            std::make_shared<ov::opset1::Convert>(new_scale_param, scale_convert->get_destination_type());
        new_scale_convert->set_friendly_name(scale_convert->get_friendly_name() + "/expert_" +
                                             std::to_string(ctx.expert_idx));
        scale_for_multiply = new_scale_convert->output(0);
    } else {
        scale_for_multiply = new_scale_param->output(0);
    }

    // 5. Multiply: scale * weights
    auto new_multiply = std::make_shared<ov::opset1::Multiply>(scale_for_multiply, weights_for_multiply);
    new_multiply->set_friendly_name(ctx.multiply_node->get_friendly_name() + "/expert_" +
                                    std::to_string(ctx.expert_idx));

    // 6. Convert after Multiply if needed
    if (ctx.convert_after_multiply) {
        auto new_convert_after_multiply =
            std::make_shared<ov::opset1::Convert>(new_multiply, ctx.convert_after_multiply->get_destination_type());
        new_convert_after_multiply->set_friendly_name(ctx.convert_after_multiply->get_friendly_name() + "/expert_" +
                                                      std::to_string(ctx.expert_idx));
        return new_convert_after_multiply->output(0);
    }
    return new_multiply->output(0);
}

/**
 * @brief Helper to prepare input branches based on detected pattern
 */
inline ov::OutputVector prepare_input_branches(ov::Output<ov::Node> matmul_input0,
                                               size_t num_branches,
                                               const std::string& matmul_name) {
    ov::OutputVector branches;

    // Pattern 1: Check for Reshape → Tile → Convert (batched pattern)
    if (auto reshape_node = std::dynamic_pointer_cast<ov::opset1::Reshape>(matmul_input0.get_node_shared_ptr())) {
        auto tile_node =
            std::dynamic_pointer_cast<ov::opset1::Tile>(reshape_node->input_value(0).get_node_shared_ptr());
        if (tile_node) {
            auto convert_input =
                std::dynamic_pointer_cast<ov::opset1::Convert>(tile_node->input_value(0).get_node_shared_ptr());
            if (convert_input) {
                LOG_INFO("  Detected Pattern 1: Batched (Reshape→Tile→Convert)");

                // Create shared Convert and Reshape for all branches
                auto input_param_source = convert_input->input_value(0);
                auto shared_convert =
                    std::make_shared<ov::opset1::Convert>(input_param_source, convert_input->get_destination_type());
                shared_convert->set_friendly_name(convert_input->get_friendly_name() + "/shared");

                // Modify reshape shape: change first dim from num_experts to 1
                auto orig_reshape_const =
                    std::dynamic_pointer_cast<ov::opset1::Constant>(reshape_node->input_value(1).get_node_shared_ptr());
                auto orig_shape_vec = orig_reshape_const->cast_vector<int64_t>();
                std::vector<int64_t> new_shape_vec = orig_shape_vec;
                new_shape_vec[0] = 1;

                auto new_reshape_const = std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                                                ov::Shape{new_shape_vec.size()},
                                                                                new_shape_vec);
                auto shared_reshape =
                    std::make_shared<ov::opset1::Reshape>(shared_convert->output(0), new_reshape_const, false);
                shared_reshape->set_friendly_name(reshape_node->get_friendly_name() + "/shared");

                // All branches use the same shared reshape
                for (size_t i = 0; i < num_branches; ++i) {
                    branches.push_back(shared_reshape->output(0));
                }
                return branches;
            }
        }
    }

    // Pattern 2: Check for Concat
    if (auto concat = std::dynamic_pointer_cast<ov::opset1::Concat>(matmul_input0.get_node_shared_ptr())) {
        LOG_INFO("  Detected Pattern 2: Concat with " << concat->get_input_size() << " branches");
        return concat->input_values();
    }

    // Pattern 3: Sliceable input - create Slice operations
    LOG_INFO("  Detected Pattern 3: Sliceable input, creating " << num_branches << " Slice operations");

    auto input0_shape = matmul_input0.get_partial_shape();
    if (!input0_shape.rank().is_static()) {
        LOG_ERROR("  Input0 shape rank is not static for slicing");
        return branches;
    }

    for (size_t i = 0; i < num_branches; ++i) {
        std::vector<int64_t> start_vec = {static_cast<int64_t>(i)};
        std::vector<int64_t> stop_vec = {static_cast<int64_t>(i + 1)};
        std::vector<int64_t> step_vec = {1};
        std::vector<int64_t> axes_vec = {0};

        auto start_const = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{1}, start_vec);
        auto stop_const = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{1}, stop_vec);
        auto step_const = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{1}, step_vec);
        auto axes_const = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{1}, axes_vec);

        auto slice_op =
            std::make_shared<ov::op::v8::Slice>(matmul_input0, start_const, stop_const, step_const, axes_const);
        slice_op->set_friendly_name(matmul_name + "/slice_input0_expert_" + std::to_string(i));
        branches.push_back(slice_op->output(0));
    }

    return branches;
}

}  // anonymous namespace

// =============================================================================
// UnrollMoEMatMul
// =============================================================================
// Unified pass for unrolling MoE expert MatMul patterns
// Automatically detects and handles three input patterns:
//   Pattern 1 (Batched):   input_param → convert → tile → reshape → MatMul
//   Pattern 2 (Concat):    Concat([a,b,c,d]) → MatMul
//   Pattern 3 (Sliceable): AnyInput[N,...] → (auto-sliced) → MatMul
// All patterns share: scale_param + weights_param → multiply → convert → MatMul (input1)
//
// Transforms to N expert branches with individual parameters and Concat output

UnrollMoEMatMul::UnrollMoEMatMul(std::shared_ptr<ov::Model> model) : model_(model) {
    MATCHER_SCOPE(UnrollMoEMatMul);

    auto matmul_pattern = ov::pass::pattern::wrap_type<ov::opset1::MatMul>();

    auto callback = [this](ov::pass::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ov::opset1::MatMul>(m.get_match_root());
        if (!matmul)
            return false;

        LOG_INFO("UnrollMoEMatMul: Checking MatMul " << matmul->get_friendly_name());

        auto matmul_input0 = matmul->input_value(0);
        auto matmul_input1 = matmul->input_value(1);

        // ========== Step 1: Check input1 (weights path - common to all patterns) ==========
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

        auto scale_param = get_param_node(scale_param_source);
        auto weights_param = get_param_node(weights_param_source);

        if (!scale_param || !weights_param) {
            LOG_DEBUG("  Could not find scale or weights parameter nodes");
            return false;
        }

        // Get parameter shapes to determine num_experts
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

        if (scale_num_experts != weights_num_experts) {
            LOG_DEBUG("  Number of experts mismatch: scale=" << scale_num_experts
                                                             << ", weights=" << weights_num_experts);
            return false;
        }

        // Auto-detect num_experts from parameter shapes
        size_t num_experts = scale_num_experts;

        LOG_INFO("  Found MoE MatMul pattern with " << num_experts << " experts (auto-detected)");

        // ========== Step 2: Prepare input0 branches (pattern-specific) ==========
        auto input0_branches = prepare_input_branches(matmul_input0, num_experts, matmul->get_friendly_name());

        if (input0_branches.empty()) {
            LOG_ERROR("  Failed to prepare input branches");
            return false;
        }

        if (input0_branches.size() != num_experts) {
            LOG_DEBUG("  Input branches count " << input0_branches.size() << " != num_experts " << num_experts
                                                << ", skipping");
            return false;
        }

        // ========== Step 3: Create per-expert parameters ==========
        auto scale_shape_vec = scale_orig_shape.to_shape();
        auto weights_shape_vec = weights_orig_shape.to_shape();

        ov::Shape scale_new_shape = scale_shape_vec;
        ov::Shape weights_new_shape = weights_shape_vec;
        scale_new_shape[0] = 1;
        weights_new_shape[0] = 1;

        LOG_INFO("  Scale: " << scale_shape_vec << " → " << scale_new_shape);
        LOG_INFO("  Weights: " << weights_shape_vec << " → " << weights_new_shape);

        ov::NodeVector expert_outputs;
        ov::ParameterVector new_params;

        // ========== Step 4: Create expert branches ==========
        for (size_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            auto input0_branch = input0_branches[expert_idx];

            // Create expert branch weights using helper function
            ExpertBranchContext ctx{expert_idx,
                                    scale_param,
                                    weights_param,
                                    scale_param_source,
                                    weights_param_source,
                                    multiply_node,
                                    convert_after_multiply,
                                    matmul,
                                    scale_new_shape,
                                    weights_new_shape};

            auto weights_for_matmul = create_expert_branch_weights(ctx, new_params);

            // MatMul: input0_branch × weights
            auto new_matmul = std::make_shared<ov::opset1::MatMul>(input0_branch,
                                                                   weights_for_matmul,
                                                                   matmul->get_transpose_a(),
                                                                   matmul->get_transpose_b());
            new_matmul->set_friendly_name(matmul->get_friendly_name() + "/expert_" + std::to_string(expert_idx));

            expert_outputs.push_back(new_matmul);
        }

        // ========== Step 5: Concat and replace ==========
        auto output_concat = std::make_shared<ov::opset1::Concat>(expert_outputs, 0);
        output_concat->set_friendly_name(matmul->get_friendly_name() + "/concat");

        // Register new parameters with model
        model_->add_parameters(new_params);

        // Replace original MatMul
        ov::copy_runtime_info(matmul, output_concat);
        ov::replace_node(matmul, output_concat);

        LOG_INFO("  Successfully created " << num_experts << " expert branches with " << new_params.size()
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
            // Add RTInfo for parameter mapping
            new_param->get_rt_info()["moe_original_param"] = param_node->get_friendly_name();
            new_param->get_rt_info()["moe_expert_index"] = static_cast<int64_t>(i);
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
// UnrollParameterMultiply
// =============================================================================
// Transforms: Multiply(Param[N,...], NonConcat)  =>  Concat([Multiply(Param[0], Slice(NonConcat,0)), ...])
// Handles cases like Multiply(k_parameter, Swish) where Swish is not unrolled
// The parameter must be unrolled for correct weight loading in partial unroll scenarios

UnrollParameterMultiply::UnrollParameterMultiply(std::shared_ptr<ov::Model> model) : model_(model) {
    MATCHER_SCOPE(UnrollParameterMultiply);

    auto multiply_pattern = ov::pass::pattern::wrap_type<ov::opset1::Multiply>();

    auto callback = [this](ov::pass::pattern::Matcher& m) {
        auto multiply = m.get_match_root();
        LOG_INFO("UnrollParameterMultiply: Checking Multiply " << multiply->get_friendly_name());

        auto multiply_op = std::dynamic_pointer_cast<ov::opset1::Multiply>(multiply);
        if (!multiply_op) {
            return false;
        }

        auto input0 = multiply_op->input_value(0);
        auto input1 = multiply_op->input_value(1);

        // Check if either input is a Concat - if so, skip (handled by PushMultiplyBeforeConcat)
        auto input0_concat = std::dynamic_pointer_cast<ov::opset1::Concat>(input0.get_node_shared_ptr());
        auto input1_concat = std::dynamic_pointer_cast<ov::opset1::Concat>(input1.get_node_shared_ptr());

        if (input0_concat || input1_concat) {
            LOG_DEBUG("UnrollParameterMultiply: One input is Concat, skipping (handled by PushMultiplyBeforeConcat)");
            return false;
        }

        // Try to find a parameter input (possibly through Convert)
        ov::Output<ov::Node> param_input, other_input;
        std::shared_ptr<ov::op::v0::Parameter> param_node;

        auto param0 = get_param_node(input0);
        auto param1 = get_param_node(input1);

        if (param0 && !param1) {
            param_node = param0;
            param_input = input0;
            other_input = input1;
        } else if (param1 && !param0) {
            param_node = param1;
            param_input = input1;
            other_input = input0;
        } else {
            // Either no parameter or both are parameters - skip
            return false;
        }

        // Check parameter shape: first dimension must be > 1 for unrolling
        auto param_shape = param_node->get_partial_shape();

        if (!param_shape.rank().is_static() || !param_shape[0].is_static()) {
            LOG_DEBUG("UnrollParameterMultiply: Parameter shape not static, skipping");
            return false;
        }

        int64_t num_branches = param_shape[0].get_length();

        if (num_branches <= 1) {
            LOG_DEBUG("UnrollParameterMultiply: Parameter first dimension <= 1, no need to unroll");
            return false;
        }

        // Check other input shape compatibility for slicing
        auto other_shape = other_input.get_partial_shape();

        if (!other_shape.rank().is_static() || !other_shape[0].is_static()) {
            LOG_DEBUG("UnrollParameterMultiply: Other input shape not static, cannot slice");
            return false;
        }

        int64_t other_dim0 = other_shape[0].get_length();

        if (other_dim0 != num_branches) {
            LOG_DEBUG("UnrollParameterMultiply: Shape mismatch - param[0]=" << num_branches
                                                                            << " vs other[0]=" << other_dim0);
            return false;
        }

        LOG_INFO("UnrollParameterMultiply: Unrolling Multiply with parameter");
        LOG_INFO("  Parameter: " << param_node->get_friendly_name() << ", shape: " << param_shape);
        LOG_INFO("  Other input type: " << other_input.get_node()->get_type_name());
        LOG_INFO("  Number of branches: " << num_branches);

        // Create new per-branch parameters
        auto orig_shape = param_shape.to_shape();
        ov::Shape new_param_shape = orig_shape;
        new_param_shape[0] = 1;  // Change first dimension from N to 1

        ov::NodeVector new_multiply_outputs;
        ov::ParameterVector new_params;

        // Check if there's a Convert between parameter and multiply
        auto param_source = param_input;
        std::shared_ptr<ov::opset1::Convert> param_convert;
        if (auto conv = std::dynamic_pointer_cast<ov::opset1::Convert>(param_input.get_node_shared_ptr())) {
            param_convert = conv;
        }

        for (size_t i = 0; i < static_cast<size_t>(num_branches); ++i) {
            // Create new parameter for this branch
            auto new_param = std::make_shared<ov::op::v0::Parameter>(param_node->get_element_type(),
                                                                     ov::PartialShape(new_param_shape));
            new_param->set_friendly_name(param_node->get_friendly_name() + "/branch_" + std::to_string(i));
            new_param->get_rt_info()["moe_original_param"] = param_node->get_friendly_name();
            new_param->get_rt_info()["moe_expert_index"] = static_cast<int64_t>(i);
            new_params.push_back(new_param);

            ov::Output<ov::Node> param_for_multiply = new_param->output(0);

            // Apply Convert if original had one
            if (param_convert) {
                auto new_convert =
                    std::make_shared<ov::opset1::Convert>(param_for_multiply, param_convert->get_destination_type());
                new_convert->set_friendly_name(param_convert->get_friendly_name() + "/branch_" + std::to_string(i));
                param_for_multiply = new_convert->output(0);
            }

            // Slice other input for this branch
            // Create Slice: other_input[i:i+1, ...]
            auto start = ov::opset1::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(i)});
            auto stop = ov::opset1::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(i + 1)});
            auto step = ov::opset1::Constant::create(ov::element::i64, {1}, {1});
            auto axes = ov::opset1::Constant::create(ov::element::i64, {1}, {0});

            auto slice = std::make_shared<ov::op::v8::Slice>(other_input, start, stop, step, axes);
            slice->set_friendly_name(other_input.get_node()->get_friendly_name() + "/slice_" + std::to_string(i));

            // Create Multiply for this branch
            ov::Output<ov::Node> multiply_input0, multiply_input1;
            if (param_input == multiply->input_value(0)) {
                multiply_input0 = param_for_multiply;
                multiply_input1 = slice->output(0);
            } else {
                multiply_input0 = slice->output(0);
                multiply_input1 = param_for_multiply;
            }

            auto new_multiply = std::make_shared<ov::opset1::Multiply>(multiply_input0, multiply_input1);
            new_multiply->set_friendly_name(multiply->get_friendly_name() + "/branch_" + std::to_string(i));
            new_multiply_outputs.push_back(new_multiply);
        }

        // Concat all branch outputs
        auto new_concat = std::make_shared<ov::opset1::Concat>(new_multiply_outputs, 0);
        new_concat->set_friendly_name(multiply->get_friendly_name() + "/concat");

        // Register new parameters with model
        model_->add_parameters(new_params);

        ov::copy_runtime_info(multiply, new_concat);
        ov::replace_node(multiply, new_concat);

        LOG_INFO("  Successfully unrolled Multiply into " << num_branches << " branches with " << new_params.size()
                                                          << " new parameters");
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_pattern, matcher_name);
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
