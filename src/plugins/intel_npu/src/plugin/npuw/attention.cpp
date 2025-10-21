// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "attention.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/op_types.hpp"  // is_parameter
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/validate.hpp"
#include "util.hpp"

namespace opp = ov::pass::pattern;

namespace {
enum class SDPA_Inputs : std::size_t { Q = 0, K, V, M, NUM_REQUIRED };

// Implementation of RemoveEmptyKVTensors from llm_compiled_model.cpp
class RemoveEmptyKVTensors : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::RemoveEmptyKVTensors");

    struct Context {
        std::vector<std::shared_ptr<ov::opset13::Parameter>> old_params;
        using Ref = std::reference_wrapper<Context>;
    };

    RemoveEmptyKVTensors(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        // Handle both direct parameter and parameter with convert
        auto param_or_convert =
            std::make_shared<opp::op::Or>(ov::OutputVector{param, opp::wrap_type<ov::op::v0::Convert>({param})});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({param_or_convert, opp::any_input()});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();

            // Find the parameter - it could be direct or through a convert
            std::shared_ptr<ov::op::v0::Parameter> matched_param = nullptr;
            auto first_input = matched_node_concat->input(0).get_source_output().get_node_shared_ptr();

            if (ov::is_type<ov::op::v0::Parameter>(first_input)) {
                // Direct parameter case
                matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(first_input);
            } else if (ov::is_type<ov::op::v0::Convert>(first_input)) {
                // Parameter through convert case
                auto convert_input = first_input->input(0).get_source_output().get_node_shared_ptr();
                if (ov::is_type<ov::op::v0::Parameter>(convert_input)) {
                    matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(convert_input);
                }
            }

            if (!matched_param) {
                return false;  // Pattern didn't match properly
            }

            ctx.get().old_params.push_back(matched_param);

            auto users = matched_param->get_users();
            if (users.size() == 2u) {
                auto shapeof_node = ov::is_type<ov::op::v3::ShapeOf>(users[0]) ? users[0] : users[1];
                NPUW_ASSERT(ov::is_type<ov::op::v3::ShapeOf>(shapeof_node));
                auto cst_node =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, matched_param->get_shape());
                ov::replace_node(shapeof_node, cst_node);
            } else {
                NPUW_ASSERT(users.size() == 1u);
            }

            // Redirect second concat input to every node which reads from concat
            auto curr_kv_tensor = matched_node_concat->input(1).get_source_output();
            for (auto target_input : matched_node_concat->output(0u).get_target_inputs()) {
                target_input.replace_source_output(curr_kv_tensor);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(concat, "RemoveEmptyKVTensors"), std::move(callback));
    }
};

// Implementation of remove_empty_kv_inputs from llm_compiled_model.cpp
bool remove_empty_kv_inputs(std::shared_ptr<ov::Model> model) {
    ov::pass::GraphRewrite rewr;
    RemoveEmptyKVTensors::Context ctx;
    rewr.add_matcher<RemoveEmptyKVTensors>(std::ref(ctx));
    rewr.run_on_model(model);
    for (auto old_param : ctx.old_params) {
        model->remove_parameter(old_param);
    }
    ov::pass::Validate().run_on_model(model);
    // NB: if old_params is not empty - pass has been applied
    return !ctx.old_params.empty();
}

}  // namespace

// Helper function to patch broadcast constants (set to 1 for dynamic handling)
void patch_broadcast_constants(const std::shared_ptr<ov::Model>& model, size_t target_length) {
    for (auto&& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v3::Broadcast>(op)) {
            continue;
        }
        // Inspect the constant
        auto shape_source = op->input(1).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Constant>(shape_source)) {
            LOG_WARN("SDPA Broadcast's 2nd input is not Const: " << shape_source << ", skipping");
            continue;
        }

        auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_source);
        auto shape_values = shape_const->cast_vector<int32_t>();
        for (auto&& d : shape_values) {
            //  Assume the context length is the mask's innermost dimension
            if (static_cast<std::size_t>(d) == target_length) {
                d = 1;
            }
        }
        auto new_const = std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(),
                                                                shape_const->get_shape(),
                                                                shape_values);
        op->input(1).replace_source_output(new_const);
    }
}

std::optional<ov::npuw::function::Attention> ov::npuw::function::Attention::from(
    const std::shared_ptr<ov::Model>& model) {
    ov::npuw::function::Attention dyn;

    // Find the mask input (also sizeable). FIXME: We know too much at this point
    auto ops = model->get_ordered_ops();
    auto sdpa_iter = std::find_if(ops.begin(), ops.end(), [](auto&& node_ptr) {
        return ov::is_type<ov::op::v13::ScaledDotProductAttention>(node_ptr);
    });
    if (sdpa_iter == ops.end()) {
        LOG_WARN("SDPA is not found in the attn subgraph!");
        return std::nullopt;
    }

    // Traverse the SDPA's mask input upwards to find the proper Parameter.
    // Only unary ops are allowed along the way
    auto sdpa_node = *sdpa_iter;
    NPUW_ASSERT(sdpa_node->inputs().size() >= util::_v(SDPA_Inputs::NUM_REQUIRED));

    auto mask_in_node = sdpa_node->inputs()[util::_v(SDPA_Inputs::M)].get_source_output().get_node_shared_ptr();
    while (mask_in_node && !ov::op::util::is_parameter(mask_in_node)) {
        if (mask_in_node->inputs().size() != 1) {
            LOG_WARN("Non-unary or disconnected op on the way from SDPA to input mask");
            return std::nullopt;
        }
        mask_in_node = mask_in_node->inputs()[0].get_source_output().get_node_shared_ptr();
    }
    NPUW_ASSERT(ov::op::util::is_parameter(mask_in_node));
    dyn._mask = std::static_pointer_cast<ov::op::v0::Parameter>(mask_in_node);
    dyn._mask_shape = dyn._mask->get_shape();

    // Find the attention inputs with dynamic range
    const auto& f_params = model->get_parameters();
    NPUW_ASSERT(f_params.size() > 0);

    auto find_context_dim = [&](const auto& param, auto&& f) {
        const auto& param_shape = param->get_shape();
        // Look for the dynamic parameter size - past size in this case
        // With our approach it is context_size - query_size
        auto past_len = dyn.context_len() - dyn.query_len();
        auto dim_iter = std::find(param_shape.begin(), param_shape.end(), past_len);
        if (dim_iter == param_shape.end()) {
            // No such dim found
            return false;
        }
        if (std::find(dim_iter + 1, param_shape.end(), past_len) != param_shape.end()) {
            // There must be no other such dim
            return false;
        }
        f(std::distance(param_shape.begin(), dim_iter));
        return true;
    };

    for (auto&& param : f_params) {
        // A bad test but it is what it is
        if (ov::npuw::util::starts_with(param->get_friendly_name(), "past")) {
            if (!find_context_dim(param, [&](std::size_t dim_idx) {
                    dyn._inputs.push_back(ov::npuw::function::Attention::Param{param, dim_idx});
                })) {
                LOG_WARN("Couldn't identify SDPA parameter's dynamic dimension");
                return std::nullopt;
            }
        }
    }  // for(f_params)

    // There must be exactly two inputs found, for past_k and past_v.
    if (dyn._inputs.size() != 2u || !dyn._mask) {
        return std::nullopt;
    }

    // Apply transformation to the model. Note: only function body is modified
    // Accumulate the reshape map
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
    for (auto&& p : dyn._inputs) {
        ov::PartialShape dyn_shape = p.param->get_shape();  // Here it is yet static
        dyn_shape[p.dim] = ov::Dimension();                 // ..and now is dynamic
        new_shapes[p.param->output(0)] = std::move(dyn_shape);
    }
    // Mask
    {
        ov::PartialShape dyn_shape = dyn._mask_shape;
        // Put the mask's innermost dimension dynamic
        *dyn_shape.rbegin() = ov::Dimension();
        new_shapes[dyn._mask->output(0)] = std::move(dyn_shape);
    }
    model->reshape(new_shapes);

    // Patch Broadcast constants if there's any. If there's broadcast in the attention
    // block, its shape argument is normally a precomputed Const (which would be
    // an expression/a subgraph in the original dynamic IR). Since we retrofit
    // dynamism into a static shape environment here, we need to patch it back.
    patch_broadcast_constants(model, dyn.context_len());
    model->validate_nodes_and_infer_types();

    return {std::move(dyn)};
}

ov::npuw::function::SDPAPatternNodes ov::npuw::function::findSDPAPatternNodes(const std::shared_ptr<ov::Model>& model) {
    // Find decomposed SDPA pattern components
    ov::npuw::function::SDPAPatternNodes pattern_nodes;

    // Search for the pattern: MatMul -> Add -> Softmax -> MatMul
    auto ops = model->get_ordered_ops();
    for (auto&& node : ops) {
        if (ov::is_type<ov::op::v8::Softmax>(node)) {
            pattern_nodes.softmax_node = node;

            // Check if softmax is fed by Add
            auto softmax_input = node->input(0).get_source_output().get_node_shared_ptr();
            if (ov::is_type<ov::op::v1::Add>(softmax_input)) {
                pattern_nodes.add_node = softmax_input;

                // Check if add is fed by MatMul (first MatMul)
                auto add_input0 = pattern_nodes.add_node->input(0).get_source_output().get_node_shared_ptr();
                if (ov::is_type<ov::op::v0::MatMul>(add_input0)) {
                    pattern_nodes.matmul1_node = add_input0;
                }
            }

            // Check if softmax feeds into MatMul (second MatMul)
            for (auto&& output : node->outputs()) {
                for (auto&& target_input : output.get_target_inputs()) {
                    auto target_node = target_input.get_node()->shared_from_this();
                    if (ov::is_type<ov::op::v0::MatMul>(target_node)) {
                        pattern_nodes.matmul2_node = target_node;
                        break;
                    }
                }
                if (pattern_nodes.matmul2_node)
                    break;
            }

            if (pattern_nodes.isValid()) {
                break;  // Found complete pattern
            }
        }
    }

    return pattern_nodes;
}

std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node) {
    if (!add_node || add_node->get_input_size() < 2) {
        return nullptr;
    }

    // Traverse the Add node's mask input (input 1) upwards to find the proper Parameter
    // Only unary ops are allowed along the way
    auto mask_in_node = add_node->input(1).get_source_output().get_node_shared_ptr();
    while (mask_in_node && !ov::op::util::is_parameter(mask_in_node)) {
        if (mask_in_node->inputs().size() != 1) {
            LOG_WARN("Non-unary or disconnected op on the way from Add to input mask");
            return nullptr;
        }
        mask_in_node = mask_in_node->inputs()[0].get_source_output().get_node_shared_ptr();
    }

    if (mask_in_node && ov::op::util::is_parameter(mask_in_node)) {
        return std::static_pointer_cast<ov::op::v0::Parameter>(mask_in_node);
    }

    return nullptr;
}

// Helper struct to hold validation and setup results
struct PyramidValidationResult {
    size_t query_length;
    size_t full_context_length;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;
};

// Helper struct to hold model processing result
struct PyramidModelResult {
    std::shared_ptr<ov::Model> model;
    ov::npuw::function::Attention attention;
};

// Helper function to patch reshape constants for pre-reshape (-1 substitution)
void patch_reshape_constants_pre_reshape(const std::shared_ptr<ov::Model>& model,
                                         const ov::npuw::function::Attention& dyn) {
    for (auto&& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::Reshape>(op)) {
            continue;
        }

        // Check if Reshape's single consumer is MatMul
        auto target_inputs = op->output(0).get_target_inputs();
        if (target_inputs.size() != 1) {
            continue;  // Reshape should have exactly one consumer
        }

        auto matmul_node = target_inputs.begin()->get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v0::MatMul>(matmul_node)) {
            continue;
        }

        // Check if MatMul's input 0 is from Softmax
        auto matmul_input0 = matmul_node->input(0).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v8::Softmax>(matmul_input0)) {
            continue;
        }

        LOG_INFO("Found Reshape -> MatMul pattern where MatMul input 0 is from Softmax, patching Reshape constant");

        // Inspect the reshape constant (shape input)
        auto shape_source = op->input(1).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Constant>(shape_source)) {
            LOG_WARN("Reshape's shape input is not Const: " << shape_source << ", skipping");
            continue;
        }

        auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_source);
        auto shape_values = shape_const->cast_vector<int32_t>();

        int64_t value_seq_dim = -1;
        for (auto parm : dyn._inputs) {
            auto parm_node = parm.param;
            if (ov::npuw::util::isPastKeyValuesValue(parm_node->get_friendly_name())) {
                value_seq_dim = static_cast<int64_t>(parm.dim);
            }
        }
        NPUW_ASSERT(value_seq_dim != -1);
        shape_values[value_seq_dim] = -1;

        auto new_const = std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(),
                                                                shape_const->get_shape(),
                                                                shape_values);
        op->input(1).replace_source_output(new_const);
    }
}

// Helper function to process a single pyramid model (clone, reshape, patch, optimize)
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t query_length,
                                                        size_t full_context_length,
                                                        const std::map<std::string, size_t>& past_key_sequence_dims,
                                                        const std::map<std::string, size_t>& past_value_sequence_dims) {
    // Clone the original model for modification
    auto cloned_model = original_model->clone();

    // Calculate dimensions for this model
    size_t current_context_length = (model_idx + 1) * query_length;
    size_t current_past_length = model_idx * query_length;

    LOG_DEBUG("Model " << model_idx << ":");
    LOG_DEBUG("  Context length: " << current_context_length);
    LOG_DEBUG("  Past length: " << current_past_length);

    // Create Attention instance for this model
    ov::npuw::function::Attention dyn;

    // Find SDPA pattern nodes in the cloned model
    auto cloned_pattern_nodes = ov::npuw::function::findSDPAPatternNodes(cloned_model);
    if (!cloned_pattern_nodes.isValid()) {
        LOG_WARN("Could not find SDPA pattern in cloned model " << model_idx);
        return std::nullopt;
    }

    // Find mask parameter in the cloned model
    auto cloned_mask_param = find_mask_parameter(cloned_pattern_nodes.add_node);
    if (!cloned_mask_param) {
        LOG_WARN("Could not find mask parameter in cloned model " << model_idx);
        return std::nullopt;
    }

    // Create reshape map for this model
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;

    // Update parameters shapes
    const auto& params = cloned_model->get_parameters();
    for (auto&& param : params) {
        const std::string param_name = param->get_friendly_name();
        auto original_shape = param->get_shape();
        ov::PartialShape new_shape = original_shape;

        // Handle attention mask parameter - use the mask parameter found in cloned model
        if (param == cloned_mask_param) {
            // Update the last dimension to current context length
            if (new_shape.size() >= 1) {
                new_shape[new_shape.size() - 1] = current_context_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Mask param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);
            }
        }
        // Handle past key parameters
        else if (ov::npuw::util::isPastKeyValuesKey(param_name)) {
            // Use pre-analyzed sequence dimension information
            auto dim_iter = past_key_sequence_dims.find(param_name);
            if (dim_iter != past_key_sequence_dims.end()) {
                size_t sequence_dim_idx = dim_iter->second;
                new_shape[sequence_dim_idx] = current_past_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Past key param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);

                // Record past key input in dyn
                dyn._inputs.push_back(ov::npuw::function::Attention::Param{param, sequence_dim_idx});
            } else {
                LOG_WARN("No pre-analyzed sequence dimension for past key param: " << param_name);
                return std::nullopt;
            }
        }
        // Handle past value parameters
        else if (ov::npuw::util::isPastKeyValuesValue(param_name)) {
            // Use pre-analyzed sequence dimension information
            auto dim_iter = past_value_sequence_dims.find(param_name);
            if (dim_iter != past_value_sequence_dims.end()) {
                size_t sequence_dim_idx = dim_iter->second;
                new_shape[sequence_dim_idx] = current_past_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Past value param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);

                // Record past value input in dyn
                dyn._inputs.push_back(ov::npuw::function::Attention::Param{param, sequence_dim_idx});
            } else {
                LOG_WARN("No pre-analyzed sequence dimension for past value param: " << param_name);
                return std::nullopt;
            }
        }
    }

    // Apply the reshaping to the cloned model
    if (new_shapes.empty()) {
        LOG_WARN("No parameters found for reshaping in model " << model_idx << ", skipping this model");
        return std::nullopt;
    }

    // Apply pre-reshape patching using helper functions
    patch_broadcast_constants(cloned_model, full_context_length);
    patch_reshape_constants_pre_reshape(cloned_model, dyn);

    cloned_model->reshape(new_shapes);
    cloned_model->validate_nodes_and_infer_types();

    LOG_DEBUG("Model " << model_idx << " reshaped successfully");

    // For model 0, past length is 0, so we can optimize by removing empty KV inputs
    if (model_idx == 0 && current_past_length == 0) {
        NPUW_ASSERT(remove_empty_kv_inputs(cloned_model));
    }

    // Dump the reshaped model for debugging using OpenVINO serialize pass
    std::string model_path = "pyramid_model_" + std::to_string(model_idx) + "_after_reshape.xml";
    std::string weights_path = "pyramid_model_" + std::to_string(model_idx) + "_after_reshape.bin";
    std::cout << "Dumping reshaped pyramid model " << model_idx << " to: " << model_path << std::endl;
    try {
        ov::pass::Serialize serialize_pass(model_path, weights_path);
        serialize_pass.run_on_model(cloned_model);
        std::cout << "Successfully dumped reshaped model " << model_idx << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to dump reshaped model " << model_idx << ": " << e.what() << std::endl;
    }

    auto updated_pattern_nodes = ov::npuw::function::findSDPAPatternNodes(cloned_model);
    if (!updated_pattern_nodes.isValid()) {
        LOG_WARN("Could not find updated SDPA pattern after reshape for model " << model_idx
                                                                                << ", skipping this model");
        return std::nullopt;
    }

    auto updated_mask_param = find_mask_parameter(updated_pattern_nodes.add_node);
    if (!updated_mask_param) {
        LOG_WARN("Could not find updated mask parameter after reshape for model " << model_idx
                                                                                  << ", skipping this model");
        return std::nullopt;
    }

    dyn._mask = updated_mask_param;
    dyn._mask_shape = updated_mask_param->get_shape();
    LOG_DEBUG("  Updated mask parameter after reshape");

    // Log attention information for this model
    LOG_DEBUG("  Attention info - mask: " << (dyn._mask ? "present" : "absent"));
    LOG_DEBUG("  Attention info - inputs count: " << dyn._inputs.size());
    if (dyn._mask) {
        LOG_DEBUG("  Mask shape: " << dyn._mask_shape);
    }

    return PyramidModelResult{cloned_model, std::move(dyn)};
}

// Helper function to validate model and extract necessary information for pyramid attention
std::optional<PyramidValidationResult> validate_and_setup_pyramid_attention(const std::shared_ptr<ov::Model>& model) {
    // Find SDPA pattern nodes using the extracted function
    auto pattern_nodes = ov::npuw::function::findSDPAPatternNodes(model);
    if (!pattern_nodes.isValid()) {
        return std::nullopt;
    }

    LOG_INFO("Found SDPA pattern: MatMul -> Add -> Softmax -> MatMul");

    // Extract query_length and full_context_length from Softmax output shape
    auto softmax_output_shape = pattern_nodes.softmax_node->get_output_shape(0);
    size_t query_length = 0;
    size_t full_context_length = 0;

    if (softmax_output_shape.size() >= 2) {
        full_context_length = softmax_output_shape.back();                     // Last dimension
        query_length = softmax_output_shape[softmax_output_shape.size() - 2];  // Second-to-last dimension

        LOG_DEBUG("Extracted from Softmax output shape:");
        LOG_DEBUG("  Query length: " << query_length);
        LOG_DEBUG("  Full context length: " << full_context_length);
    } else {
        LOG_WARN("Softmax output shape has insufficient dimensions: " << softmax_output_shape.size());
        return std::nullopt;
    }

    // Early return for invalid parameters
    if (query_length == 0 || full_context_length == 0 || full_context_length < query_length) {
        LOG_WARN("Invalid query_length (" << query_length << ") or full_context_length (" << full_context_length
                                          << ") for pyramid attention");
        return std::nullopt;
    }

    // Pre-analyze original model to find sequence dimensions for past key/value parameters
    // This avoids repeated analysis in each cloned model
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    // Helper function to find sequence dimension in parameter shape
    auto find_context_dim = [](const std::shared_ptr<ov::op::v0::Parameter>& param,
                               size_t target_length) -> std::optional<size_t> {
        const auto& param_shape = param->get_shape();
        auto dim_iter = std::find(param_shape.begin(), param_shape.end(), target_length);
        if (dim_iter == param_shape.end()) {
            return std::nullopt;  // No such dim found
        }
        if (std::find(dim_iter + 1, param_shape.end(), target_length) != param_shape.end()) {
            return std::nullopt;  // There must be no other such dim
        }
        return std::distance(param_shape.begin(), dim_iter);
    };

    // Analyze original model parameters to find sequence dimensions
    const auto& original_params = model->get_parameters();
    for (const auto& param : original_params) {
        const std::string param_name = param->get_friendly_name();

        if (ov::npuw::util::isPastKeyValuesKey(param_name)) {
            auto sequence_dim_opt = find_context_dim(param, full_context_length - query_length);
            if (sequence_dim_opt) {
                past_key_sequence_dims[param_name] = *sequence_dim_opt;
                LOG_DEBUG("Found past key sequence dimension for '" << param_name << "': " << *sequence_dim_opt);
            } else {
                LOG_WARN("Could not find sequence dimension for past key param: " << param_name);
                return std::nullopt;
            }
        } else if (ov::npuw::util::isPastKeyValuesValue(param_name)) {
            auto sequence_dim_opt = find_context_dim(param, full_context_length - query_length);
            if (sequence_dim_opt) {
                past_value_sequence_dims[param_name] = *sequence_dim_opt;
                LOG_DEBUG("Found past value sequence dimension for '" << param_name << "': " << *sequence_dim_opt);
            } else {
                LOG_WARN("Could not find sequence dimension for past value param: " << param_name);
                return std::nullopt;
            }
        }
    }

    return PyramidValidationResult{query_length, full_context_length, past_key_sequence_dims, past_value_sequence_dims};
}

std::optional<ov::npuw::function::PyramidAttention> ov::npuw::function::PyramidAttention::from(
    const std::shared_ptr<ov::Model>& model) {
    // Validate and setup pyramid attention
    auto validation_result = validate_and_setup_pyramid_attention(model);
    if (!validation_result) {
        return std::nullopt;
    }

    size_t query_length = validation_result->query_length;
    size_t full_context_length = validation_result->full_context_length;
    const auto& past_key_sequence_dims = validation_result->past_key_sequence_dims;
    const auto& past_value_sequence_dims = validation_result->past_value_sequence_dims;

    std::vector<std::shared_ptr<ov::Model>> pyramid_models;
    size_t num_models = full_context_length / query_length;
    LOG_INFO("Creating " << num_models << " pyramid attention models");

    // Store Attention instances for each model
    std::vector<ov::npuw::function::Attention> pyramid_attentions;

    for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
        // Process each pyramid model using the helper function
        auto result = process_pyramid_model(model,
                                            model_idx,
                                            query_length,
                                            full_context_length,
                                            past_key_sequence_dims,
                                            past_value_sequence_dims);
        if (!result) {
            return std::nullopt;
        }

        pyramid_models.push_back(result->model);
        pyramid_attentions.push_back(std::move(result->attention));
    }

    LOG_INFO("Successfully created " << pyramid_models.size() << " pyramid attention models");

    // Create PyramidAttention instance and set the extracted values
    ov::npuw::function::PyramidAttention pyramid_attention;
    pyramid_attention._query_length = query_length;
    pyramid_attention._full_context_length = full_context_length;
    pyramid_attention._models = pyramid_models;
    pyramid_attention._attentions = pyramid_attentions;

    // Early return with pyramid attention result
    LOG_INFO("Returning pyramid attention with " << pyramid_models.size() << " models");
    LOG_INFO("  Query length: " << pyramid_attention._query_length);
    LOG_INFO("  Full context length: " << pyramid_attention._full_context_length);
    LOG_INFO("  Attention instances: " << pyramid_attention._attentions.size());
    return pyramid_attention;
}

ov::npuw::runtime::attention::PositionIDs::PositionIDs(std::size_t param_idx,
                                                       const ov::npuw::compiled::Attention& d,
                                                       const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_d(d),
      m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_d.query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

ov::npuw::runtime::attention::Selector::Ptr ov::npuw::runtime::attention::PositionIDs::find(
    const ov::npuw::compiled::Attention& d,
    const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, d, rq)};
    }
    return Selector::Ptr{};
}

void ov::npuw::runtime::attention::PositionIDs::prepare(int64_t past_len) {
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();
    const auto pos_ids_len = static_cast<int64_t>(in_dims.back());

    // There's several cases possible:
    // a. Prefill input_ids, including chunk
    // b. Generate input_ids, 1
    // c. Generate input_ids, N (speculative)
    // Prefill (even chunked) is left-padded, so for (a) it's enough to take the last element.
    // Same works for b (there's no choice).
    // c may require traversing the tensor backwards as Generate with N>1 is right_padded (?)

    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = pos_ids_len - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            m_current_length = pos_data_ptr[idx];
            switch (m_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                m_past_length = m_current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                m_past_length = ((past_len + m_d.query_size - 1) / m_d.query_size) * m_d.query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;
}

int64_t ov::npuw::runtime::attention::PositionIDs::length() const {
    return m_current_length;
}

int64_t ov::npuw::runtime::attention::PositionIDs::past_length() const {
    return m_past_length;
}
