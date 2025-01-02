// Copyright (C) 2023-2024 Intel Corporation
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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace opp = ov::pass::pattern;
class TransposeValueTensors : public ov::pass::MatcherPass {
public:
    struct Context {
        std::vector<std::shared_ptr<ov::opset13::Parameter>> new_params;
        std::vector<std::shared_ptr<ov::opset13::Parameter>> old_params;
        using Ref = std::reference_wrapper<Context>;
    };

    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors");
    TransposeValueTensors(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({param, transpose});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, concat});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();

            auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_param);
            auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(matched_node_concat);
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(matched_node_transpose);
            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);

            auto shape = matched_param->get_partial_shape();
            OPENVINO_ASSERT(shape.size() == 4u);
            // NB: Transpose Parameter that correspond to V-tensor it will
            // speed-up its multiplication with attention scores
            std::swap(shape[2], shape[3]);
            auto new_param = std::make_shared<ov::opset13::Parameter>(matched_param->get_element_type(), shape);
            new_param->set_friendly_name(matched_param->get_friendly_name());
            new_param->outputs().begin()->get_tensor().set_names(
                matched_param->outputs().begin()->get_tensor().get_names());
            ov::replace_node(matched_param, new_param);
            // NB: Save in order to add/remove to the model later on
            ctx.get().new_params.push_back(new_param);
            ctx.get().old_params.push_back(matched_param);

            auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
            auto new_transpose =
                std::make_shared<ov::opset13::Transpose>(matched_transpose->input_value(0), order_cst->output(0));
            new_transpose->set_friendly_name(matched_transpose->get_friendly_name());
            ov::replace_node(matched_transpose, new_transpose);

            auto new_concat =
                std::make_shared<ov::opset13::Concat>(ov::OutputVector{new_param->output(0), new_transpose->output(0)},
                                                      3u);
            new_concat->set_friendly_name(matched_concat->get_friendly_name());
            ov::replace_node(matched_concat, new_concat);

            matched_matmul->set_transpose_b(true);

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors"), std::move(callback));
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

std::shared_ptr<ov::Model> cvt_value_tensors_layout(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor ppp(model);
    for (auto tensor : model->outputs()) {
        if (tensor.get_any_name().find("value") != std::string::npos) {
            // NB: [batch, num_heads, seq_len, emb_size] -> [batch, num_heads, emb_size, seq_len]
            ppp.output(tensor.get_any_name()).model().set_layout(ov::Layout("BHSE"));
            ppp.output(tensor.get_any_name()).tensor().set_layout(ov::Layout("BHES"));
        }
    }
    return ppp.build();
}

bool optimize_value_tensors(std::shared_ptr<ov::Model> model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ScaledDotProductAttentionDecomposition>();
    TransposeValueTensors::Context ctx;
    rewr.add_matcher<TransposeValueTensors>(std::ref(ctx));
    rewr.run_on_model(model);

    model->add_parameters(ctx.new_params);
    for (auto old_param : ctx.old_params) {
        model->remove_parameter(old_param);
    }
    ov::pass::Validate().run_on_model(model);

    // NB: if new_params is not empty - pass has been applied
    return !ctx.new_params.empty();
}

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

KVAxesPosition get_kv_axes(const std::string& model_type) {
    KVAxesPosition axes;
    if (model_type == "chatglm") {
        axes.batch = 1u;
        axes.seq_len = 0u;
    } else if (model_type == "qwen") {
        // Note, qwen2 does not fall into this category and conforms to default layout
        axes.batch = 0u;
        axes.seq_len = 1u;
    } else {
        axes.batch = 0u;
        axes.seq_len = 2u;
    }
    return axes;
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
};

std::optional<NPUDesc> extract_npu_descriptor(const std::shared_ptr<const ov::IPlugin>& plugin) {
    const ov::Any arch = plugin->get_property(ov::device::architecture.name(), ov::AnyMap{});
    const ov::Any max_tiles = plugin->get_property(ov::intel_npu::max_tiles.name(), ov::AnyMap{});
    return std::make_optional(NPUDesc{arch.as<std::string>(), max_tiles.as<int64_t>()});
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

ov::AnyMap get_baseline_common_config() {
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
    return config;
}

ov::AnyMap get_default_common_config(const std::shared_ptr<ov::Model>& model) {
    auto config = get_baseline_common_config();
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    return config;
}

ov::AnyMap get_default_prefill_config(const std::shared_ptr<ov::Model>& model, const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(model);
    if (is_cw_compressed(model)) {
        config.emplace("NPUW_DQ", "YES");
    } else {
        config.emplace("NPUW_PMM", "NO");
    }
    if (npudesc.has_value() && npudesc->arch == "4000" && npudesc->max_tiles != -1) {
        config.emplace("NPU_DPU_GROUPS", npudesc->max_tiles);
    }
    return config;
}

ov::AnyMap get_default_generate_config(const std::shared_ptr<ov::Model>& model,
                                       const std::optional<NPUDesc>& npudesc,
                                       const ::intel_npu::npuw::llm::GenerateHint hint) {
    auto config = get_default_common_config(model);
    if (hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF) {
        config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    }
    // NB: Unconditionally set for generation model
    config.emplace("NPUW_DQ", "YES");
    if (npudesc.has_value() && npudesc->arch == "4000") {
        config.emplace("NPU_DPU_GROUPS", 4);
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
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    LOG_DEBUG("Creating LLMCompiledModel");
    LOG_BLOCK();

    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);

    std::map<std::string, ov::Any> npuw_llm_props;
    std::map<std::string, ov::Any> other_props;
    split_llm_properties(properties, npuw_llm_props, other_props);

    // Remove "NPUW_LLM_PREFILL_CONFIG", "NPUW_LLM_GENERATE_CONFIG" from map,
    // to not pass them into ::intel_npu::Config object, as we don't need to
    // preserve them somewhere.
    auto prefill_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_GENERATE_CONFIG"));

    m_cfg.update(any_copy(npuw_llm_props));

    LOG_DEBUG("1. Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();
    LOG_DEBUG("2. Transform kvcache model from stateful to stateless.");
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);
    LOG_DEBUG("3. Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    const ::intel_npu::npuw::llm::ModelDesc model_desc = m_cfg.get<::intel_npu::NPUW_LLM_MODEL_DESC>();
    const uint32_t kMaxPromptLen = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t kMinResponseLen = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);
    KVAxesPosition axes = get_kv_axes(model_desc.type);
    m_kvcache_desc = KVCacheDesc{kMaxPromptLen, kMaxPromptLen + kMinResponseLen, 0u, axes.seq_len};
    LOG_DEBUG("4. Make prefill model with static shapes");
    reshape_to_static(prefill_model, m_kvcache_desc.max_prompt_size, m_kvcache_desc.max_prompt_size, axes);
    LOG_DEBUG("5. Make kvcache model with static shapes");
    reshape_to_static(kvcache_model, 1u, m_kvcache_desc.total_size, axes);
    LOG_DEBUG("6.Check and apply opt layout if applicable.");
    // NB: Try to apply opt transpose only for Llama-2-7b-chat-hf model
    if (model_desc.name_or_path == "meta-llama/Llama-2-7b-chat-hf" ||
        (model_desc.type == "llama" && model_desc.num_key_value_heads == 32)) {
        if (optimize_value_tensors(kvcache_model)) {
            // NB: Check if TransposeValueTensors transformation was applied
            m_kvcache_desc.v_tensors_transposed = true;
            prefill_model = cvt_value_tensors_layout(prefill_model);
        }
    }
    LOG_DEBUG("7. Optimize kvcache model to output key/values for new token.");
    kvcache_model = redirect_new_kv_to_output(kvcache_model);
    LOG_DEBUG("8. Converting KV-cache in kvcache model to FP16.");
    kvcache_model = cvt_kvcache_to_fp16(kvcache_model);
    LOG_DEBUG("9. Converting KV-cache in prefill model to FP16.");
    prefill_model = cvt_kvcache_to_fp16(prefill_model);

    auto npudesc = extract_npu_descriptor(plugin);
    auto prefill_config =
        prefill_config_opt.value_or(get_default_prefill_config(prefill_model, npudesc)).as<ov::AnyMap>();

    const ::intel_npu::npuw::llm::GenerateHint generate_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_HINT>();
    LOG_DEBUG("9. Passed GENERATE_HINT: " << std::string(::intel_npu::NPUW_LLM_GENERATE_HINT::toString(generate_hint)));
    // NB: GENERATE_HINT is only applicable for default generate config!
    if (generate_config_opt.has_value() && npuw_llm_props.count(ov::intel_npu::npuw::llm::generate_hint.name())) {
        OPENVINO_THROW("GENERATE_HINT is only applicable for default generate config!");
    }
    auto generate_config =
        generate_config_opt.value_or(get_default_generate_config(model, npudesc, generate_hint)).as<ov::AnyMap>();

    merge_config_with(prefill_config, other_props);
    merge_config_with(generate_config, other_props);

    m_kvcache_compiled = std::make_shared<ov::npuw::CompiledModel>(kvcache_model, plugin, generate_config);
    m_prefill_compiled = std::make_shared<ov::npuw::CompiledModel>(prefill_model, plugin, prefill_config);

    implement_properties();
    LOG_DEBUG("Done");
}

void ov::npuw::LLMCompiledModel::export_model(std::ostream& model) const {
    OPENVINO_NOT_IMPLEMENTED;
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
                          BIND(npuw::llm::model_desc, NPUW_LLM_MODEL_DESC, getString),
                          BIND(npuw::llm::max_prompt_len, NPUW_LLM_MAX_PROMPT_LEN, get),
                          BIND(npuw::llm::min_response_len, NPUW_LLM_MIN_RESPONSE_LEN, get),
                          BIND(npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT, getString)});
#undef BIND
}
