// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "transformations/convert_precision.hpp"
#include "util.hpp"

namespace opp = ov::pass::pattern;

class RemoveEmptyKVTensors : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::RemoveEmptyKVTensors");

    struct Context {
        std::vector<std::shared_ptr<ov::opset13::Parameter>> old_params;
        using Ref = std::reference_wrapper<Context>;
    };

    RemoveEmptyKVTensors(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto concat = opp::wrap_type<ov::op::v0::Concat>({param, opp::any_input()});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_param = ov::as_type_ptr<ov::op::v0::Parameter>(node_to_output.at(param).get_node_shared_ptr());
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();

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

class TransposeValueTensors : public ov::pass::MatcherPass {
public:
    struct Context {
        using Ref = std::reference_wrapper<Context>;
        bool bTransposed = false;
    };

protected:
    // generic part of matchers, to transpose v-tensors, and concat, and update matmul args
    void transpose_matmul_b(Context::Ref ctx,
                            std::shared_ptr<ov::Node> node_param,
                            std::shared_ptr<ov::Node> node_concat,
                            std::shared_ptr<ov::Node> node_transpose,
                            std::shared_ptr<ov::Node> node_matmul) {
        auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(node_param);
        auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(node_concat);
        auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(node_transpose);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(node_matmul);

        auto param_shape = matched_param->get_partial_shape();
        NPUW_ASSERT(param_shape.size() == 4u);
        // NB: Transpose Parameter that correspond to V-tensor it will
        // speed-up its multiplication with attention scores
        std::swap(param_shape[2], param_shape[3]);

        matched_param->set_partial_shape(param_shape);

        auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});

        matched_transpose->set_argument(1, order_cst);
        matched_concat->set_axis(3u);
        matched_matmul->set_transpose_b(true);
        ctx.get().bTransposed = true;
    }
};

// llama2 pattern for value tensor concate
class TransposeValueTensors_llama2 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama2");
    TransposeValueTensors_llama2(Context::Ref ctx) {
        register_matcher_llama2(ctx);
    }

private:
    void register_matcher_llama2(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, concat});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();

            transpose_matmul_b(ctx,
                               matched_node_param,
                               matched_node_concat,
                               matched_node_transpose,
                               matched_node_matmul);
            LOG_DEBUG("vtensors transposed: LLama2 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama2"), std::move(callback));
    }
};

// llama3, phi3, mistral, etc, concate value tensors with broadcasting
class TransposeValueTensors_llama3 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama3");
    TransposeValueTensors_llama3(Context::Ref ctx) {
        register_matcher_llama3(ctx);
    }

private:
    void register_matcher_llama3(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});

        // only difference is that broadcast wrapped into unsquese/reshape, while transposed tensor didn't change
        const auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({concat, unsqueeze_axes});
        auto broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({unsqueeze, opp::any_input()});
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({broadcast, opp::any_input()});

        // v8 softmax? what? can be other softmaxes
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, reshape});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            auto matched_node_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
            auto matched_node_unsqueeze_axes = node_to_output.at(unsqueeze_axes).get_node_shared_ptr();
            auto matched_node_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
            auto matched_node_reshape = node_to_output.at(reshape).get_node_shared_ptr();

            auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_param);
            auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(matched_node_concat);
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(matched_node_transpose);
            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(matched_node_unsqueeze);
            auto matched_broadcast = std::static_pointer_cast<ov::op::v3::Broadcast>(matched_node_broadcast);
            auto matched_reshape = std::static_pointer_cast<ov::op::v1::Reshape>(matched_node_reshape);

            auto shape_broadcast = matched_broadcast->get_output_shape(0);
            NPUW_ASSERT(shape_broadcast.size() == 5u);
            std::swap(shape_broadcast[3], shape_broadcast[4]);

            LOG_DEBUG("shape_broadcast for: " << matched_broadcast->get_friendly_name()
                                              << ", shape=" << shape_broadcast);

            const auto broadcast_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, shape_broadcast);
            broadcast_axes_node->set_friendly_name(matched_broadcast->get_friendly_name() + "/new_broadcast_shape");
            matched_broadcast->input(1).replace_source_output(broadcast_axes_node);

            auto shape_reshape = matched_reshape->get_output_shape(0);
            NPUW_ASSERT(shape_reshape.size() == 4u);
            std::swap(shape_reshape[2], shape_reshape[3]);

            LOG_DEBUG("shape_reshape for: " << matched_reshape->get_friendly_name() << ", shape=" << shape_reshape);

            const auto reshape_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, shape_reshape);
            reshape_axes_node->set_friendly_name(matched_reshape->get_friendly_name() + "/new_reshape_shape");
            matched_reshape->input(1).replace_source_output(reshape_axes_node);

            transpose_matmul_b(ctx,
                               matched_node_param,
                               matched_node_concat,
                               matched_node_transpose,
                               matched_node_matmul);
            LOG_DEBUG("vtensors transposed: LLama3 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama3"), std::move(callback));
    }
};

class ScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::ScaledDotProductAttentionDecomposition");
    ScaledDotProductAttentionDecomposition() {
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            auto new_output_node = decompose(node);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, "ScaledDotProductAttentionDecomposition");
        register_matcher(m, std::move(callback));
    }
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node) {
        using namespace ov::op;
        using namespace ov;
        auto query = node->input_value(0);
        auto key = node->input_value(1);
        auto value = node->input_value(2);
        auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
        auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
        auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
        auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
        auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
        auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

        Output<Node> scale;
        if (node->get_input_size() < 5) {
            scale = register_new_node<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
            scale = register_new_node<v1::ConvertLike>(scale, query);
            auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
            scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
        } else {
            scale = node->input_value(4);
        }

        auto q_scaled = register_new_node<v1::Multiply>(query, scale);
        auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
        auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
        auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
        k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
        auto minus_inf =
            register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
                ->output(0);
        auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
        auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

        auto scaled_atten = register_new_node<v0::MatMul>(q_scaled, key, false, true)->output(0);
        minus_inf = register_new_node<v1::ConvertLike>(minus_inf, scaled_atten);

        if (node->get_causal() || node->get_input_size() > 3) {
            Output<Node> mask;
            Output<Node> atten_mask;
            if (!node->get_causal()) {
                mask = node->input_value(3);

                // two types of masks are supported. A boolean mask where a value of True indicates that the element
                // should take part in attention. A float mask of the same type as query, key, value that is added to
                // the attention score.
                if (mask.get_element_type() == element::boolean) {
                    atten_mask = register_new_node<v1::ConvertLike>(mask, scaled_atten);
                    auto inv_mask = register_new_node<v1::LogicalNot>(mask);
                    atten_mask = register_new_node<v1::Select>(inv_mask, atten_mask, minus_inf);
                } else {
                    atten_mask = mask;
                }
            } else {
                auto target_s_len = register_new_node<v8::Gather>(q_shape, minus_two, zero_i);
                auto source_s_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
                auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
                auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
                auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
                mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
                auto horizontal_range =
                    register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
                horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
                auto stop = register_new_node<v1::Add>(target_s_len, one_i);
                auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
                vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
                auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
                atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
            }
            scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
        }

        scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
        auto result = register_new_node<v0::MatMul>(scaled_atten, value);
        result->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, get_new_nodes());
        return result;
    }
};

namespace {
uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::shared_ptr<ov::Model> cvt_kvcache_to_fp16(const std::shared_ptr<ov::Model>& model) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (const auto& tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    for (const auto& tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    return ppp.build();
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (std::size_t i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
    model->validate_nodes_and_infer_types();
    return model;
}

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

// testability - should we have interface for that or move matchers to another file?
bool optimize_value_tensors(std::shared_ptr<ov::Model> model);
bool optimize_value_tensors(std::shared_ptr<ov::Model> model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ScaledDotProductAttentionDecomposition>();
    TransposeValueTensors::Context ctx;
    rewr.add_matcher<TransposeValueTensors_llama2>(std::ref(ctx));
    rewr.add_matcher<TransposeValueTensors_llama3>(std::ref(ctx));
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);

    // NB: matmul parameters gets transposed, if pass applied
    return ctx.bTransposed;
}

namespace {
struct KVAxesPosition {
    uint32_t batch;
    uint32_t seq_len;
};

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const KVAxesPosition& kv_axes_position) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("inputs_embeds") != std::string::npos) {
            // NB: VLMs case, model accepts inputs_embeds[BATCH, SEQ_LEN, EMB_SIZE]
            NPUW_ASSERT(input.get_partial_shape().size() == 3u);
            NPUW_ASSERT(input.get_partial_shape()[2].is_static());
            new_shape = ov::PartialShape({1, input_size, input.get_partial_shape()[2]});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

bool is_cw_compressed(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> rt_info_path = {"nncf", "weight_compression", "group_size"};
    if (!model->has_rt_info(rt_info_path)) {
        // NB: Model isn't compressed by NNCF - skip
        return false;
    }
    auto group_size = model->get_rt_info<int>(rt_info_path);
    if (group_size == -1) {
        // NB: Enable DQ for CW quantized models
        return true;
    }
    return false;
}

struct NPUDesc {
    std::string arch;
    int64_t max_tiles;
    bool compiler_dq;
};

std::optional<NPUDesc> extract_npu_descriptor(const std::shared_ptr<const ov::IPlugin>& plugin) {
    const auto all_devices = plugin->get_core()->get_property("NPU", ov::available_devices);
    if (all_devices.empty()) {
        return std::nullopt;
    }

    const std::string arch = plugin->get_property(ov::device::architecture.name(), ov::AnyMap{}).as<std::string>();
    const int64_t max_tiles = plugin->get_property(ov::intel_npu::max_tiles.name(), ov::AnyMap{}).as<int64_t>();
    bool compiler_dq = false;
    // Don't use reference here!
    const auto supported_properties =
        plugin->get_property(ov::supported_properties.name(), ov::AnyMap{}).as<std::vector<ov::PropertyName>>();
    if (std::find(supported_properties.begin(), supported_properties.end(), "NPU_COMPILER_DYNAMIC_QUANTIZATION") !=
        supported_properties.end()) {
        compiler_dq = true;
    }
    return std::make_optional(NPUDesc{arch, max_tiles, compiler_dq});
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

ov::AnyMap get_baseline_common_config(const std::optional<NPUDesc>& npudesc) {
    ov::AnyMap config = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm"},
        {"NPUW_DEVICES", "NPU"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_DCOFF_TYPE", "f16"},
        {"NPUW_DCOFF_SCALE", "YES"},
        {"NPUW_WEIGHTS_BANK", "shared"},
        {"NPUW_SLICE_OUT", "YES"},
        {"NPUW_FUNCALL_ASYNC", "YES"}};
    // FIXME: this config logic is getting more and more complex
    if (npudesc.has_value() && npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
        config.emplace("NPUW_DQ_FULL", "NO");
        config.emplace("NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES");
        config.erase("NPUW_DCOFF_TYPE");
        config.erase("NPUW_DCOFF_SCALE");
    }
    return config;
}

ov::AnyMap get_default_common_config(const std::shared_ptr<ov::Model>& model, const std::optional<NPUDesc>& npudesc) {
    auto config = get_baseline_common_config(npudesc);
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    return config;
}

ov::AnyMap get_default_prefill_config(const std::shared_ptr<ov::Model>& model,
                                      const std::optional<NPUDesc>& npudesc,
                                      const ::intel_npu::npuw::llm::PrefillHint hint) {
    auto config = get_default_common_config(model, npudesc);
    if (hint == ::intel_npu::npuw::llm::PrefillHint::DYNAMIC) {
        if (is_cw_compressed(model)) {
            // NB: These two ifs are not combined into one with && deliberately
            // there may be later changes w.r.t. required compiler versions for GQ
            config.emplace("NPUW_ONLINE_PIPELINE", "SPATIAL");
        }
    }
    if (npudesc.has_value() && npudesc->arch == "4000" && npudesc->max_tiles != -1) {
        config.emplace("NPU_DPU_GROUPS", npudesc->max_tiles);
    }
    // Specify NPUW DQ if Compiler DQ is not enabled
    if (!npudesc.has_value() || !npudesc->compiler_dq) {
        if (is_cw_compressed(model)) {
            config.emplace("NPUW_DQ", "YES");
        } else {
            config.emplace("NPUW_PMM", "NO");
        }
    }
    return config;
}

ov::AnyMap get_default_generate_config(const std::shared_ptr<ov::Model>& model,
                                       const std::optional<NPUDesc>& npudesc,
                                       const ::intel_npu::npuw::llm::GenerateHint hint) {
    auto config = get_default_common_config(model, npudesc);
    if (hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF) {
        config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    }
    if (npudesc.has_value() && npudesc->arch == "4000") {
        config.emplace("NPU_DPU_GROUPS", 4);
    }
    if (hint == ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE) {
        config.emplace("NPUW_UNFOLD_IREQS", "YES");
    }
    // Specify NPUW DQ if Compiler DQ is not enabled
    if (!npudesc.has_value() || !npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
    }
    return config;
}

void merge_config_with(ov::AnyMap& lhs, const ov::AnyMap& rhs) {
    for (const auto& [key, value] : rhs) {
        // NB: Overwrite the value if key already exists
        if (auto it = lhs.find(key); it != lhs.end()) {
            it->second = value;
        } else {
            lhs.emplace(key, value);
        }
    }
}

void split_llm_properties(const ov::AnyMap& properties, ov::AnyMap& llm_properties, ov::AnyMap& other_properties) {
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW_LLM") != it->first.npos) {
            llm_properties.insert(*it);
        } else {
            other_properties.insert(*it);
        }
    }
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}
}  // namespace

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    LOG_DEBUG("Creating LLMCompiledModel");
    LOG_BLOCK();

    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);

    ov::AnyMap npuw_llm_props;
    ov::AnyMap other_props;
    split_llm_properties(properties, npuw_llm_props, other_props);
    // Solely used for serialization at the moment
    m_non_llm_props = other_props;

    // Remove "NPUW_LLM_PREFILL_CONFIG", "NPUW_LLM_GENERATE_CONFIG" from map,
    // to not pass them into ::intel_npu::Config object, as we don't need to
    // preserve them somewhere.
    auto prefill_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_GENERATE_CONFIG"));
    auto prefill_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_GENERATE_CONFIG"));

    m_cfg.update(any_copy(npuw_llm_props));

    LOG_DEBUG("1. Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();
    LOG_DEBUG("2. Transform kvcache model from stateful to stateless.");
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);
    LOG_DEBUG("   ...also convert BF16 to FP16");
    ov::pass::ConvertPrecision(ov::element::bf16, ov::element::f16).run_on_model(kvcache_model);
    LOG_DEBUG("3. Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    const uint32_t batch_dim = m_cfg.get<::intel_npu::NPUW_LLM_BATCH_DIM>();
    const uint32_t seq_len_dim = m_cfg.get<::intel_npu::NPUW_LLM_SEQ_LEN_DIM>();
    KVAxesPosition axes{batch_dim, seq_len_dim};
    const uint32_t max_prompt_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t min_response_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);

    m_kvcache_desc = KVCacheDesc{max_prompt_len, max_prompt_len + min_response_len, 0u, seq_len_dim};
    LOG_DEBUG("4. Make prefill model with static shapes");
    reshape_to_static(prefill_model, m_kvcache_desc.max_prompt_size, m_kvcache_desc.max_prompt_size, axes);
    LOG_DEBUG("5. Make kvcache model with static shapes");
    reshape_to_static(kvcache_model, 1u, m_kvcache_desc.total_size, axes);

    const bool optimize_v_tensors = m_cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_V_TENSORS>();
    if (optimize_v_tensors) {
        LOG_DEBUG("6. Check and apply opt layout");
        LOG_BLOCK();
        if (optimize_value_tensors(kvcache_model)) {
            NPUW_ASSERT(optimize_value_tensors(prefill_model));
            m_kvcache_desc.v_tensors_transposed = true;
        } else {
            LOG_DEBUG("vtensors optimisation not applied");
        }
    } else {
        LOG_DEBUG("6. Check and apply opt layout --- SKIPPED");
    }
    NPUW_ASSERT(remove_empty_kv_inputs(prefill_model));
    LOG_DEBUG("7. Optimize kvcache model to output key/values for new token.");
    kvcache_model = redirect_new_kv_to_output(kvcache_model);
    LOG_DEBUG("8. Converting KV-cache in kvcache model to FP16.");
    kvcache_model = cvt_kvcache_to_fp16(kvcache_model);
    LOG_DEBUG("9. Converting KV-cache in prefill model to FP16.");
    prefill_model = cvt_kvcache_to_fp16(prefill_model);

    auto npudesc = extract_npu_descriptor(plugin);

    // NB: PREFILL_HINT is only applicable for default prefill config!
    if (prefill_config_opt.has_value() && npuw_llm_props.count(ov::intel_npu::npuw::llm::prefill_hint.name())) {
        OPENVINO_THROW("PREFILL_HINT only works with default prefill config!");
    }
    const ::intel_npu::npuw::llm::PrefillHint prefill_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_HINT>();
    auto prefill_config =
        prefill_config_opt.value_or(get_default_prefill_config(prefill_model, npudesc, prefill_hint)).as<ov::AnyMap>();

    // NB: GENERATE_HINT is only applicable for default generate config!
    if (generate_config_opt.has_value() && npuw_llm_props.count(ov::intel_npu::npuw::llm::generate_hint.name())) {
        OPENVINO_THROW("GENERATE_HINT only works with default generate config!");
    }
    const ::intel_npu::npuw::llm::GenerateHint generate_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_HINT>();
    auto generate_config =
        generate_config_opt.value_or(get_default_generate_config(kvcache_model, npudesc, generate_hint))
            .as<ov::AnyMap>();

    const auto& prefill_config_addition_value =
        prefill_config_addition.has_value() ? prefill_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};
    const auto& generate_config_addition_value =
        generate_config_addition.has_value() ? generate_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};

    merge_config_with(prefill_config, other_props);
    merge_config_with(generate_config, other_props);
    merge_config_with(prefill_config, prefill_config_addition_value);
    merge_config_with(generate_config, generate_config_addition_value);

    m_kvcache_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(kvcache_model, plugin, generate_config));
    NPUW_ASSERT(m_kvcache_compiled && "Can't create ov::npuw::CompiledModel for passed kvcache "
                                      "model and its config, please check passed config.");
    m_prefill_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(prefill_model, plugin, prefill_config));
    NPUW_ASSERT(m_prefill_compiled && "Can't create ov::npuw::CompiledModel for passed prefill "
                                      "model and its config, please check passed config.");

    implement_properties();
    LOG_DEBUG("Done");
}

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const bool serialized)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    NPUW_ASSERT(serialized && "This constructor should only be utilized during deserialization!");
    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);
    LOG_DEBUG("LLMCompiledModel is being deserialized, skipping the full constructor flow...");
}

void ov::npuw::LLMCompiledModel::export_model(std::ostream& stream) const {
    using namespace ov::npuw::s11n;

    // Identify encryption flow
    bool encryption_required = false;
    EncryptionCallbacks enc_callbacks;
    if (auto it = m_non_llm_props.find(ov::cache_encryption_callbacks.name());
        it != m_non_llm_props.end() && it->second.as<EncryptionCallbacks>().encrypt) {
        LOG_INFO("Encryption will be done via the function provided.");
        encryption_required = true;
        enc_callbacks.encrypt = it->second.as<EncryptionCallbacks>().encrypt;
    }

    // Identify either full flow or weightless
    bool is_weightless = true;
    if (auto it = m_non_llm_props.find(ov::cache_mode.name());
        it != m_non_llm_props.end() && it->second.as<CacheMode>() == CacheMode::OPTIMIZE_SPEED) {
        LOG_INFO("Serialization will be done via flow with weights.");
        is_weightless = false;
    }

    // Write header regardless of encryption requirement - to identify NPUW serializated blobs
    // Serialize magic number first
    write(stream, NPUW_SERIALIZATION_INDICATOR);
    // Serialize general meta info
    write(stream, OPENVINO_VERSION_MAJOR);
    write(stream, OPENVINO_VERSION_MINOR);
    write(stream, OPENVINO_VERSION_PATCH);
    write(stream, std::string(NPUW_SERIALIZATION_VERSION));
    // Serialize encrypted flag
    write(stream, encryption_required);
    // Write flow identifier
    write(stream, is_weightless);

    if (!encryption_required) {
        LLMSerializeContext ctx(false, nullptr);
        return serialize(stream, ctx);
    }

    // In case of weightless flow the whole blob will be encrypted on NPUW side.
    std::stringstream non_encrypted_stream;
    if (is_weightless) {
        non_encrypted_stream.copyfmt(stream);
        LLMSerializeContext ctx(false, nullptr);
        serialize(non_encrypted_stream, ctx);
        std::string encrypted = enc_callbacks.encrypt(non_encrypted_stream.str());
        write(stream, encrypted);
    } else {
        // In case of blob with weights only encrypt XML part of the model
        LLMSerializeContext ctx(true, enc_callbacks.encrypt);
        serialize(stream, ctx);
    }
}

void ov::npuw::LLMCompiledModel::serialize(std::ostream& stream, const ov::npuw::s11n::LLMSerializeContext& ctx) const {
    LOG_INFO("Serializing LLMCompiledModel...");
    LOG_BLOCK();

    using namespace ov::npuw::s11n;

    // Identify either full flow or weightless
    bool is_weightless = true;
    if (auto it = m_non_llm_props.find(ov::cache_mode.name());
        it != m_non_llm_props.end() && it->second.as<CacheMode>() == CacheMode::OPTIMIZE_SPEED) {
        LOG_INFO("Serialization will be done via flow with weights.");
        is_weightless = false;
    }

    auto write_model_meta = [&](std::ostream& model_stream) {
        // Serialize name
        write(model_stream, m_name);

        // Serialize inputs and outputs
        write(model_stream, inputs());
        write(model_stream, outputs());

        // Serialize LLMCompiledModel-specific data
        write(model_stream, m_kvcache_desc.max_prompt_size);
        write(model_stream, m_kvcache_desc.total_size);
        write(model_stream, m_kvcache_desc.num_stored_tokens);
        write(model_stream, m_kvcache_desc.dim);
        write(model_stream, m_kvcache_desc.v_tensors_transposed);

        // Write config
        write(model_stream, m_cfg);

        // Serialize CompiledModels
        m_kvcache_compiled->serialize(model_stream);
        m_prefill_compiled->serialize(model_stream);
    };

    std::stringstream non_encrypted_stream;
    if (ctx.encrypted) {
        NPUW_ASSERT(ctx.encrypt && "Encryption function isn't provided!");
        non_encrypted_stream.copyfmt(stream);
        write_model_meta(non_encrypted_stream);
        std::string encrypted_str = ctx.encrypt(non_encrypted_stream.str());
        write(stream, encrypted_str);
    } else {
        write_model_meta(stream);
    }

    // Serialize bank name
    const auto& kv_bank = m_kvcache_compiled->m_weights_bank;
    const auto& p_bank = m_prefill_compiled->m_weights_bank;
    NPUW_ASSERT(kv_bank && p_bank && kv_bank == p_bank && "Prefill and KVCache models' weight bank should be shared!");
    write(stream, kv_bank->get_name());

    if (!is_weightless) {
        // Serialize weights bank
        // Note: no need to encrypt weights in full flow
        kv_bank->serialize(stream);
    }

    LOG_INFO("Done.");
}

std::shared_ptr<ov::npuw::LLMCompiledModel> ov::npuw::LLMCompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Deserializing LLMCompiledModel...");
    LOG_BLOCK();

    using namespace ov::npuw::s11n;

    // Sanity check magic number
    ov::npuw::s11n::IndicatorType serialization_indicator;
    read(stream, serialization_indicator);
    NPUW_ASSERT(serialization_indicator == NPUW_SERIALIZATION_INDICATOR && "This blob wasn't serialized via NPUW!");

    // Deserialize general meta info
    int vmajor, vminor, vpatch;
    std::string s11n_version;
    read(stream, vmajor);
    read(stream, vminor);
    read(stream, vpatch);
    read(stream, s11n_version);

    if (vmajor != OPENVINO_VERSION_MAJOR || vminor != OPENVINO_VERSION_MINOR || vpatch != OPENVINO_VERSION_PATCH ||
        s11n_version != std::string(NPUW_SERIALIZATION_VERSION)) {
        OPENVINO_THROW("This blobs was serialized with different OV version!",
                       "\nSerialized by OV ",
                       vmajor,
                       '.',
                       vminor,
                       '.',
                       vpatch,
                       "\nCurrent OV version ",
                       OPENVINO_VERSION_MAJOR,
                       '.',
                       OPENVINO_VERSION_MINOR,
                       '.',
                       OPENVINO_VERSION_PATCH,
                       "\nNPUW serialized by version ",
                       s11n_version,
                       "\nNPUW current serialization version ",
                       NPUW_SERIALIZATION_VERSION);
    }

    bool encrypted = false;
    read(stream, encrypted);
    bool is_weightless = true;
    read(stream, is_weightless);

    auto read_and_finalize_banks = [&](std::istream& model_stream,
                                       const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled) {
        // Deserialize weights bank name
        std::string bank_name;
        read(model_stream, bank_name);

        if (is_weightless) {
            auto bank = ov::npuw::weights::bank(bank_name, compiled->get_plugin()->get_core(), "");

            compiled->m_kvcache_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->m_weights_bank = bank;

            compiled->m_kvcache_compiled->finalize_weights_bank();
            compiled->m_prefill_compiled->finalize_weights_bank();
        } else {
            auto bank =
                ov::npuw::weights::Bank::deserialize(model_stream, compiled->get_plugin()->get_core(), bank_name);

            compiled->m_kvcache_compiled->m_weights_bank = bank;
            compiled->m_prefill_compiled->m_weights_bank = bank;

            compiled->m_kvcache_compiled->reconstruct_closure();
            compiled->m_prefill_compiled->reconstruct_closure();
        }
    };

    if (!encrypted) {
        LLMDeserializeContext ctx(false, nullptr);
        auto compiled_model = ov::npuw::LLMCompiledModel::deserialize(stream, plugin, properties, ctx);
        NPUW_ASSERT(compiled_model && "Couldn't import NPUW compiled model!");
        read_and_finalize_banks(stream, compiled_model);
        LOG_INFO("Done.");
        return compiled_model;
    }

    EncryptionCallbacks enc_callbacks;
    NPUW_ASSERT(properties.count(ov::cache_encryption_callbacks.name()) &&
                properties.at(ov::cache_encryption_callbacks.name()).as<EncryptionCallbacks>().decrypt &&
                "Model is encrypted but no decrypt function was provided!");
    enc_callbacks.decrypt = properties.at(ov::cache_encryption_callbacks.name()).as<EncryptionCallbacks>().decrypt;

    LOG_INFO("Decryption will be done via the function provided.");

    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled_model = nullptr;

    // Model is encrypted
    if (is_weightless) {
        std::string encrypted_str;
        read(stream, encrypted_str);
        std::istringstream decrypted_stream(std::move(enc_callbacks.decrypt(encrypted_str)));
        LLMDeserializeContext ctx(false, nullptr);
        compiled_model = ov::npuw::LLMCompiledModel::deserialize(decrypted_stream, plugin, properties, ctx);
    } else {
        LLMDeserializeContext ctx(true, enc_callbacks.decrypt);
        compiled_model = ov::npuw::LLMCompiledModel::deserialize(stream, plugin, properties, ctx);
    }

    NPUW_ASSERT(compiled_model && "Couldn't import NPUW compiled model!");
    read_and_finalize_banks(stream, compiled_model);

    LOG_INFO("Done.");

    return compiled_model;
}

std::shared_ptr<ov::npuw::LLMCompiledModel> ov::npuw::LLMCompiledModel::deserialize(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties,
    const ov::npuw::s11n::LLMDeserializeContext& ctx) {
    using namespace ov::npuw::s11n;

    auto read_model_meta = [&](std::istream& model_stream) {
        // Deserialize model name first
        std::string model_name;
        read(model_stream, model_name);

        // Create a dummy CompiledModel with an empty ov::Model - this will skip the constructor flow
        // to continue deserialization
        ov::ParameterVector parameters;
        ov::NodeVector results;

        read(model_stream, parameters);
        read(model_stream, results);

        auto ov_model = std::make_shared<ov::Model>(results, parameters, model_name);

        auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(ov_model, plugin, true);

        // Deserialize LLMCompiledModel-specific data
        read(model_stream, compiled->m_kvcache_desc.max_prompt_size);
        read(model_stream, compiled->m_kvcache_desc.total_size);
        read(model_stream, compiled->m_kvcache_desc.num_stored_tokens);
        read(model_stream, compiled->m_kvcache_desc.dim);
        read(model_stream, compiled->m_kvcache_desc.v_tensors_transposed);

        // Deserialize config
        read(model_stream, compiled->m_cfg);
        compiled->implement_properties();

        // Deserialize CompiledModels
        compiled->m_kvcache_compiled = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties);
        compiled->m_prefill_compiled = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties);

        return compiled;
    };

    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled = nullptr;
    if (ctx.encrypted) {
        std::string encrypted_string;
        read(stream, encrypted_string);
        std::istringstream decrypted_stream(std::move(ctx.decrypt(encrypted_string)));
        compiled = read_model_meta(decrypted_stream);
    } else {
        compiled = read_model_meta(stream);
    }

    NPUW_ASSERT(compiled && "Couldn't create NPUW compiled model!");

    return compiled;
}

std::shared_ptr<const ov::Model> ov::npuw::LLMCompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::LLMCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::npuw::LLMCompiledModel::get_property(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (name == ov::intel_npu::npuw::llm::prefill_config.name() ||
        name == ov::intel_npu::npuw::llm::generate_config.name()) {
        OPENVINO_THROW(name, " is write-only option!");
    }

    auto&& configIterator = m_prop_to_opt.find(name);
    if (configIterator != m_prop_to_opt.cend()) {
        return std::get<1>(configIterator->second)(m_cfg);
    } else {
        return m_prefill_compiled->get_property(name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_sync_infer_request() const {
    auto* non_const_this = const_cast<ov::npuw::LLMCompiledModel*>(this);  // because of const in API
    return non_const_this->create_llm_infer_request();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_llm_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr);
}

void ov::npuw::LLMCompiledModel::implement_properties() {
#define BIND(N, T, GETTER)                                                                 \
    {                                                                                      \
        ov::intel_npu::N.name(), {                                                         \
            ov::PropertyMutability::RW, [](const ::intel_npu::Config& config) -> ov::Any { \
                return config.GETTER<::intel_npu::T>();                                    \
            }                                                                              \
        }                                                                                  \
    }

    m_prop_to_opt.insert({BIND(npuw::llm::enabled, NPUW_LLM, get),
                          BIND(npuw::llm::batch_dim, NPUW_LLM_BATCH_DIM, get),
                          BIND(npuw::llm::batch_dim, NPUW_LLM_SEQ_LEN_DIM, get),
                          BIND(npuw::llm::max_prompt_len, NPUW_LLM_MAX_PROMPT_LEN, get),
                          BIND(npuw::llm::min_response_len, NPUW_LLM_MIN_RESPONSE_LEN, get),
                          BIND(npuw::llm::optimize_v_tensors, NPUW_LLM_OPTIMIZE_V_TENSORS, get),
                          BIND(npuw::llm::prefill_hint, NPUW_LLM_PREFILL_HINT, getString),
                          BIND(npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT, getString)});
#undef BIND
}
