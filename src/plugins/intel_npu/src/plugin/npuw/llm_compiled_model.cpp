// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "embedding/embedding_infer_request.hpp"
#include "embedding/prepare_embedding_model.hpp"
#include "embedding/redirect_new_kv_to_output.hpp"
#include "embedding/remove_empty_kv_inputs.hpp"
#include "llm_compiled_model_utils.hpp"
#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "moe_transformations/apply_moe_device_routed_transforms.hpp"
#include "npuw_transformations/convert_kvcache_to_precision.hpp"
#include "npuw_transformations/decompose_gqa.hpp"
#include "npuw_transformations/lora_stateful_to_stateless.hpp"
#include "npuw_transformations/optimize_value_tensors.hpp"
#include "npuw_transformations/patch_phi3_sliding_mask.hpp"
#include "npuw_transformations/reshape_sliced_head_to_static.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "npuw_transformations/slice_out_embeds.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "partitioning/patterns/moe.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "partitioning/patterns/sdpa.hpp"
#include "serialization.hpp"
#include "transformations/convert_precision.hpp"
#include "util.hpp"
#include "whisper/prepare_whisper_model.hpp"
#include "whisper/whisper_infer_request.hpp"

namespace opp = ov::pass::pattern;

namespace {
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
T align_to(T value, T alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
bool is_aligned_to(T value, T alignment) {
    return value % alignment == 0;
}

}  // namespace

class CutLMHead : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CutLMHead");
    explicit CutLMHead(std::shared_ptr<ov::Model>& lm_head_model) {
        // We are interested at first input to MatMul as a cut point
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});

        // There are several patterns for matmul we are looking for:
        // Matmul -> Result
        // Matmul -> Add -> Result
        auto matmul_add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
        // Matmul -> Transpose -> Result
        auto matmul_transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
        //  Matmul -> Convert -> Result
        auto matmul_convert = opp::wrap_type<ov::op::v0::Convert>({matmul});
        // MatMul -> Divide -> Tanh -> Multiply -> Result
        auto div = opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Divide>({matmul, opp::any_input()});
        auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
        auto matmul_multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});

        auto last_op = std::make_shared<opp::op::Or>(ov::OutputVector{matmul->output(0),
                                                                      matmul_add->output(0),
                                                                      matmul_transpose->output(0),
                                                                      matmul_convert->output(0),
                                                                      matmul_multiply->output(0)});
        auto res = opp::wrap_type<ov::op::v0::Result>({last_op->output(0)});

        auto callback = [=, &lm_head_model](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            std::shared_ptr<ov::Node> matched_node_last_op = nullptr;
            if (node_to_output.count(matmul_add)) {
                matched_node_last_op = node_to_output[matmul_add].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_transpose)) {
                matched_node_last_op = node_to_output[matmul_transpose].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_convert)) {
                matched_node_last_op = node_to_output[matmul_convert].get_node_shared_ptr();
            } else if (node_to_output.count(matmul_multiply)) {
                matched_node_last_op = node_to_output[matmul_multiply].get_node_shared_ptr();
            } else {
                matched_node_last_op = matched_node_matmul;
            }
            auto matched_node_result = node_to_output.at(res).get_node_shared_ptr();

            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_result = std::static_pointer_cast<ov::op::v0::Result>(matched_node_result);

            // Some LLMs add intermediate hidden state outputs that can interfere with LM head detection.
            // Skip Result nodes that were manually added (marked with "manually_added_output" in RT_INFO).
            // For example, Eagle-3 target/draft models add "last_hidden_state" output which should be skipped.
            const auto& rt_info = matched_result->get_rt_info();
            if (rt_info.count("manually_added_output")) {
                return false;
            }

            // Cut point:
            auto matmul_first_source = matched_matmul->input(0).get_source_output();

            // Cut original model:
            matched_result->input(0).replace_source_output(matmul_first_source);
            // FIXME: Somehow for KVCache model result output gets renamed in
            //        ICompiledModel::ICompiledModel().
            //        As a WA, setting the same name to output from MatMul
            //        avoids the issue.
            matmul_first_source.set_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_result->output(0).set_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_result->validate_and_infer_types();

            // Create an additional model after cut point:
            auto new_param = std::make_shared<ov::op::v0::Parameter>(matmul_first_source.get_element_type(),
                                                                     matmul_first_source.get_partial_shape());
            new_param->output(0).add_names({ov::npuw::LLMCompiledModel::output_embeds});
            matched_matmul->input(0).replace_source_output(new_param);
            auto new_result = std::make_shared<ov::op::v0::Result>(matched_node_last_op);
            lm_head_model =
                std::make_shared<ov::Model>(ov::OutputVector{new_result->output(0)}, ov::ParameterVector{new_param});

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(res, "CutLMHead"), std::move(callback));
    }
};

namespace {
std::shared_ptr<ov::Model> cut_lm_head(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    std::shared_ptr<ov::Model> lm_head_model = nullptr;
    rewr.add_matcher<CutLMHead>(lm_head_model);
    rewr.run_on_model(model);
    if (lm_head_model) {
        lm_head_model->set_friendly_name(model->get_friendly_name() + "_lm_head");
    }
    model->validate_nodes_and_infer_types();

    return lm_head_model;
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

bool is_int8_compressed(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> rt_info_path = {"nncf", "weight_compression", "mode"};
    if (!model->has_rt_info(rt_info_path)) {
        // NB: Model isn't compressed by NNCF - skip
        return false;
    }
    auto mode = model->get_rt_info<std::string>(rt_info_path);
    if (mode.find("int8") != std::string::npos) {
        return true;
    }
    return false;
}

struct NPUDesc {
    std::string arch;
    int64_t max_tiles = 0;
    bool compiler_dq = false;
    bool compiler_matmul_gate = false;
    int64_t compiler_ver = 0;
    bool support_flash_attention_tile = false;
};

std::optional<NPUDesc> extract_npu_descriptor(const std::shared_ptr<const ov::IPlugin>& plugin,
                                              const ov::AnyMap& config) {
    if (!plugin->get_core()) {
        return std::nullopt;
    }
    std::vector<std::string> all_devices;
    try {
        all_devices = plugin->get_core()->get_property("NPU", ov::available_devices);
    } catch (const ov::Exception& ex) {
        LOG_WARN("Failed to query NPU capabilities, defaulting LLM config to backend-agnostic path: " << ex.what());
        return std::nullopt;
    }
    if (all_devices.empty()) {
        return std::nullopt;
    }

    NPUDesc desc;
    desc.arch = plugin->get_property(ov::device::architecture.name(), ov::AnyMap{}).as<std::string>();
    desc.max_tiles = plugin->get_property(ov::intel_npu::max_tiles.name(), ov::AnyMap{}).as<int64_t>();

    // Don't use reference here!
    const auto supported_properties =
        plugin->get_property(ov::supported_properties.name(), ov::AnyMap{}).as<std::vector<ov::PropertyName>>();
    if (std::find(supported_properties.begin(), supported_properties.end(), "NPU_COMPILER_DYNAMIC_QUANTIZATION") !=
        supported_properties.end()) {
        desc.compiler_dq = true;
    }

    // Get compiler version based on NPU_COMPILER_TYPE configuration
    // If NPU_COMPILER_TYPE is not specified in config, use default compiler version
    auto compiler_type_it = config.find(ov::intel_npu::compiler_type.name());
    if (compiler_type_it == config.end()) {
        // NPU_COMPILER_TYPE is not specified in config, use default compiler version
        desc.compiler_ver = plugin->get_property(ov::intel_npu::compiler_version.name(), ov::AnyMap{}).as<int64_t>();
    } else {
        // NPU_COMPILER_TYPE is specified in config, get compiler version for the specified compiler type
        auto target_compiler_type = compiler_type_it->second.as<std::string>();
        desc.compiler_ver = plugin
                                ->get_property(ov::intel_npu::compiler_version.name(),
                                               ov::AnyMap{{ov::intel_npu::compiler_type.name(), target_compiler_type}})
                                .as<int64_t>();
    }
    LOG_INFO("Compiler version: " << ONEAPI_VERSION_MAJOR(desc.compiler_ver) << "."
                                  << ONEAPI_VERSION_MINOR(desc.compiler_ver));

    constexpr std::string_view compiler_gate_support_msg =
        "Compiler: accurate gated matmul (MatMul -> Divide -> Tanh -> Multiply -> Result) : ";

    if (desc.compiler_ver >= ONEAPI_MAKE_VERSION(7, 28)) {
        // accuracy for gated matmul fixed at 7.28
        desc.compiler_matmul_gate = true;
        LOG_INFO(compiler_gate_support_msg << "supported");
    } else {
        LOG_WARN(compiler_gate_support_msg << "unsupported");
    }

    if (desc.arch == "5010" && desc.compiler_ver >= ONEAPI_MAKE_VERSION(7, 29)) {
        // Flash attention tile is supported starting from compiler version 7.29 on NPU5010
        desc.support_flash_attention_tile = true;
    }

    return std::make_optional(std::move(desc));
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

void apply_weights_bank_name(ov::AnyMap& config, const std::string& bank_name) {
    auto it = config.find("NPUW_WEIGHTS_BANK");
    if (it != config.end()) {
        if (it->second.as<std::string>().empty()) {
            NPUW_ASSERT(false && "NPUW_WEIGHTS_BANK is empty in the provided config! Please use non-empty name to "
                                 "share the model weights.");
        }
    } else {
        config["NPUW_WEIGHTS_BANK"] = bank_name;
    }
}

ov::AnyMap get_baseline_common_config(const std::optional<NPUDesc>& npudesc) {
    ov::AnyMap config = {
        {"NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm"},
        {"NPUW_DEVICES", "NPU"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_FOLD", "YES"},
        {"NPUW_DCOFF_TYPE", "f16"},
        {"NPUW_DCOFF_SCALE", "YES"},
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

    // default value is ON
    // for compiler versions >= 7.28 value is ON
    // for other compiler versions value is OFF
    if (npudesc.has_value()) {
        config.emplace("NPUW_MM_GATED", (npudesc->compiler_matmul_gate ? "YES" : "NO"));
    }
    return config;
}

ov::AnyMap get_default_common_config(const std::optional<NPUDesc>& npudesc) {
    // FIXME: add `if_model_contain_slice()` condition for `SLICE_OUT` option.
    auto config = get_baseline_common_config(npudesc);
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    return config;
}

ov::AnyMap get_default_prefill_config(const std::shared_ptr<ov::Model>& model, const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(npudesc);
    if (npudesc.has_value() && npudesc->arch == "4000" && npudesc->max_tiles != -1) {
        config.emplace("NPU_TILES", npudesc->max_tiles);
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

ov::AnyMap get_default_generate_config(const std::optional<NPUDesc>& npudesc,
                                       const ::intel_npu::npuw::llm::GenerateHint hint) {
    auto config = get_default_common_config(npudesc);
    if (hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF) {
        config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
    }
    if (hint == ::intel_npu::npuw::llm::GenerateHint::FAST_COMPILE) {
        config.emplace("NPUW_UNFOLD_IREQS", "YES");
    }
    // Specify NPUW DQ if Compiler DQ is not enabled
    if (!npudesc.has_value() || !npudesc->compiler_dq) {
        config.emplace("NPUW_DQ", "YES");
    }
    // We don't need slice out for kv cache model, especially for speculative decoding which need
    // to generate more than 1 token for each inference
    config.erase("NPUW_SLICE_OUT");
    return config;
}

ov::AnyMap get_default_lm_head_config(const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(npudesc);
    config.erase("NPUW_SLICE_OUT");
    config.erase("NPUW_FUNCALL_ASYNC");
    config.emplace("NPUW_ONLINE_PIPELINE", "NONE");
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
        if (it->first.find("NPUW_LLM") != it->first.npos || it->first.find("NPUW_WHISPER") != it->first.npos) {
            llm_properties.insert(*it);
        } else {
            other_properties.insert(*it);
        }
    }
}

void refine_dynamic_props(ov::AnyMap& llm_properties, const std::optional<NPUDesc>& npudesc) {
    if (!npudesc) {
        // No NPU device detected - no idea about the actual capabilities.
        return;
    }

    if (llm_properties.count(ov::intel_npu::npuw::llm::prefill_chunk_size.name())) {
        // The chunk size value is enforced by the config, keep it
        return;
    }

    if (npudesc->compiler_ver < ONEAPI_MAKE_VERSION(7, 22)) {
        // Specify larger chunk size for older compiler versions
        LOG_VERB("Default the prefill chunk size to 1024");
        llm_properties["NPUW_LLM_PREFILL_CHUNK_SIZE"] = 1024;
    }
}

void update_config_for_whisper(ov::AnyMap& config) {
    config.erase("NPUW_SLICE_OUT");
}

void disable_ws_for_whisper(ov::AnyMap& config) {
    config.erase("NPUW_FUNCALL_FOR_ALL");
    config.erase("NPUW_FOLD");
    config.erase("NPUW_CWAI");
}

void update_config_for_text_embed(ov::AnyMap& config) {
    config.erase("NPUW_SLICE_OUT");
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

// Detect if the model is a Mixture-of-Experts (MoE) architecture
// by checking if any node name matches MoE patterns: layers.*.mlp.router or layers.*.mlp.experts
bool is_moe_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        const std::string& node_name = op->get_friendly_name();
        // Check for MoE-specific patterns:
        // - layers.*.mlp.router (router network for expert selection)
        // - layers.*.mlp.experts (expert networks)
        // Note: "expert" also matches "experts" (plural)
        if (node_name.find(ov::npuw::patterns::moe::MLP_ROUTER_NAME) != std::string::npos ||
            node_name.find(ov::npuw::patterns::moe::MLP_EXPERT_NAME) != std::string::npos) {
            LOG_INFO("Detected MoE model: found node with MoE pattern - " << node_name);
            return true;
        }
    }
    LOG_DEBUG("Non-MoE model detected: no .mlp.router or .mlp.expert nodes found");
    return false;
}

// Apply MoE-specific configuration based on hint
void apply_moe_config(ov::AnyMap& stage_config,
                      ::intel_npu::npuw::llm::MoEHint moe_hint,
                      const std::string& stage_name) {
    if (moe_hint == ::intel_npu::npuw::llm::MoEHint::HOST_ROUTED) {
        LOG_INFO("MoE config for " << stage_name << " stage: HOST_ROUTED (host-side expert routing)");
        // MoE expert and router pattern isolation options
        const ov::AnyMap expert_opts = {
            {"NPUW_ONLINE_PIPELINE", "REP"},
            {"NPUW_ONLINE_ISOLATE", "MOE"},
            {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
            {"NPUW_UNFOLD_IREQS", "NO"},
        };
        merge_config_with(stage_config, expert_opts);
    } else if (moe_hint == ::intel_npu::npuw::llm::MoEHint::DEVICE_ROUTED) {
        if (stage_name == "PREFILL") {
            NPUW_ASSERT(false && "MoE DEVICE_ROUTED is not supported for PREFILL stage. "
                                 "DEVICE_ROUTED mode uses in-graph gather-based expert selection which is only "
                                 "optimized for GENERATE stage. Please use HOST_ROUTED or DENSE for PREFILL.");
        }
        stage_config["NPUW_UNFOLD_IREQS"] = "NO";
    } else if (moe_hint == ::intel_npu::npuw::llm::MoEHint::DENSE) {
        LOG_INFO("MoE config for " << stage_name << " stage: DENSE (all experts active)");
        // DENSE mode requires CPU-only device due to extremely long NPU compilation time and high resource consumption
        auto npuw_devices =
            stage_config.count("NPUW_DEVICES") ? stage_config.at("NPUW_DEVICES").as<std::string>() : "NPU";
        NPUW_ASSERT(npuw_devices == "CPU" &&
                    "MoE DENSE mode requires CPU-only device (NPUW_DEVICES must be 'CPU'). "
                    "DENSE activates all experts simultaneously, causing extremely long NPU compilation time. "
                    "Please set NPUW_DEVICES to 'CPU'.");
    }
}

// Apply DEVICE_ROUTED MoE transformations to models
ov::element::Type choose_kv_cache_storage_type(const std::shared_ptr<ov::Model>& model,
                                               const ::intel_npu::Config& cfg,
                                               ov::AnyMap& other_props) {
    auto kv_kache_storage_type = ov::element::f16;

    // kv-cache-precision changes to fp8 does make sense unconditionally only if LPT passes succesfully applied
    if (cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_FP8>()) {
        kv_kache_storage_type = ov::npuw::optimize_kv_cache_storage(model);
    }

    auto kv_cache_precision_hint = pop_option(other_props, ov::hint::kv_cache_precision.name());
    // ov::kv_cache_precision hint can additionally change kv-cache precision, but it might lead to less accurate
    // results
    if (kv_cache_precision_hint.has_value()) {
        auto suggested_kv_cache_precision = kv_cache_precision_hint.value().as<ov::element::Type>();
        if (kv_kache_storage_type != suggested_kv_cache_precision) {
            LOG_WARN("KV-cache precision HINT: " << suggested_kv_cache_precision << " applied");
            kv_kache_storage_type = suggested_kv_cache_precision;
        }
    }

    return kv_kache_storage_type;
}

std::shared_ptr<ov::Model> check_and_cut_lm_head(const std::shared_ptr<ov::Model>& m, const ::intel_npu::Config& cfg) {
    bool shared_head_enabled = cfg.get<::intel_npu::NPUW_LLM_SHARED_HEAD>();
    std::shared_ptr<ov::Model> lm_head_model = nullptr;
    if (shared_head_enabled) {
        LOG_DEBUG("Trying to separate Vocabulary matrix multiplication op into additional model...");
        lm_head_model = cut_lm_head(m);
        if (lm_head_model) {
            LOG_INFO("Three-model pipeline will be created: LM head will be shared between prefill and generate.");
        } else {
            LOG_WARN("Three-model pipeline is requested, but LM head cutting is failed,"
                     " two-model pipeline will be created!");
        }
    } else {
        LOG_INFO("Two-model pipeline will be created.");
    }

    return lm_head_model;
}

}  // namespace

// Apply DEVICE_ROUTED MoE transformations to models
std::vector<std::shared_ptr<ov::Model>> ov::npuw::LLMCompiledModel::create_generate_model_variants(
    const std::shared_ptr<ov::Model>& generate_model,
    const KVAxesPosition& axes,
    const uint32_t whisper_lhs_seq_size) {
    const uint32_t total_kv_size = m_kvcache_desc.total_size;
    const uint32_t min_response_len = total_kv_size - m_kvcache_desc.max_prompt_size;
    const uint32_t max_generation_token_len = m_kvcache_desc.max_generation_token_len;
    const bool enable_generate_pyramid = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_PYRAMID>();

    // Check if generate pyramid feature is enabled
    if (enable_generate_pyramid) {
        LOG_INFO("Generate pyramid feature is ENABLED");
        LOG_INFO(
            "Creating multiple generate model variants with stepping: 1K+min_response_len, 2K+min_response_len, etc.");

        // Determine KV cache size steps: (1K + min_response_len), (2K + min_response_len), (4K + min_response_len), (8K
        // + min_response_len), etc.
        std::vector<uint32_t> kv_size_steps;
        for (uint32_t base_size = 1024; base_size + min_response_len <= total_kv_size; base_size *= 2) {
            kv_size_steps.push_back(base_size + min_response_len);
        }
        // Always include the total size if it's not already in the list
        if (kv_size_steps.empty() || kv_size_steps.back() < total_kv_size) {
            kv_size_steps.push_back(total_kv_size);
        }

        LOG_DEBUG("KV cache size variants: ");
        for (const auto& size : kv_size_steps) {
            LOG_DEBUG("  - " << size);
        }

        // Store the sizes for runtime selection
        m_kvcache_sizes = std::move(kv_size_steps);
    } else {
        LOG_INFO("KV cache variants feature is DISABLED - using single model");
        // Use only the total size (traditional single-model approach)
        m_kvcache_sizes = {total_kv_size};
    }

    // Create generate model variants
    LOG_INFO("Creating " << m_kvcache_sizes.size() << " generate model variants...");
    std::vector<std::shared_ptr<ov::Model>> generate_model_variants;
    generate_model_variants.reserve(m_kvcache_sizes.size());

    for (size_t i = 0; i < m_kvcache_sizes.size(); ++i) {
        const uint32_t kv_size = m_kvcache_sizes[i];

        auto generate_variant = (kv_size == total_kv_size) ? generate_model : generate_model->clone();
        LOG_DEBUG("Variant " << (i + 1) << "/" << m_kvcache_sizes.size() << " (size=" << kv_size
                             << "): reshaping to static");

        // Reshape to target size
        ov::npuw::ReshapeToStatic(max_generation_token_len, kv_size, axes, m_max_lora_rank, whisper_lhs_seq_size)
            .run_on_model(generate_variant);

        // Set unique name for this variant
        generate_variant->set_friendly_name(generate_model->get_friendly_name() + "_kv" + std::to_string(kv_size));
        generate_model_variants.push_back(generate_variant);
    }
    LOG_INFO("Created all generate model variants");

    return generate_model_variants;
}

std::shared_ptr<ov::npuw::ICompiledModel_v0> ov::npuw::LLMCompiledModel::make_compiled_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    return std::dynamic_pointer_cast<ov::npuw::ICompiledModel_v0>(
        ov::npuw::ICompiledModel::create(model, plugin, properties));
}

void ov::npuw::LLMCompiledModel::compile_generate_model_variants(
    const std::vector<std::shared_ptr<ov::Model>>& generate_model_variants,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& generate_config) {
    // Compile multiple generate model variants with different sizes
    LOG_INFO("Compiling " << m_kvcache_sizes.size() << " generate model variants...");
    m_generate_compiled_variants.reserve(m_kvcache_sizes.size());

    for (size_t i = 0; i < m_kvcache_sizes.size(); ++i) {
        const uint32_t kv_size = m_kvcache_sizes[i];
        LOG_DEBUG("Compiling generate variant " << (i + 1) << "/" << m_kvcache_sizes.size()
                                                << " with size: " << kv_size);

        // Use the already prepared variant model
        auto& generate_variant = generate_model_variants[i];

        // Compile the variant
        auto compiled_variant = m_compiled_model_factory(generate_variant, plugin, generate_config);
        NPUW_ASSERT(compiled_variant && "Can't create ov::npuw::CompiledModel for generate variant!");

        m_generate_compiled_variants.push_back(compiled_variant);
        LOG_DEBUG("Successfully compiled generate variant with size: " << kv_size);
    }

    // Keep the original compiled model for backward compatibility (using the largest size)
    m_kvcache_compiled = m_generate_compiled_variants.back();
}

ov::npuw::LLMCompiledModel::LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties,
                                             CompiledModelFactory factory)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc),
      m_compiled_model_factory(std::move(factory)) {
    LOG_DEBUG("Creating LLMCompiledModel");
    LOG_BLOCK();
    ::intel_npu::registerNPUWLLMOptions(*m_options_desc);

    ov::AnyMap npuw_llm_props;
    ov::AnyMap other_props;
    split_llm_properties(properties, npuw_llm_props, other_props);
    const auto npudesc = extract_npu_descriptor(plugin, other_props);
    auto use_eagle_key = pop_option(other_props, std::string("NPUW_EAGLE"));

    // Remove map-valued section configs before m_cfg.update(any_copy(...)), since Config expects string options.
    auto prefill_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_GENERATE_CONFIG"));
    auto prefill_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_PREFILL_CONFIG"));
    auto generate_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_GENERATE_CONFIG"));
    // Also make these maps for third: lm head model, in case it will be created:
    auto lm_head_config_opt = pop_option(npuw_llm_props, std::string("NPUW_LLM_SHARED_HEAD_CONFIG"));
    auto lm_head_config_addition = pop_option(npuw_llm_props, std::string("++NPUW_LLM_SHARED_HEAD_CONFIG"));

    m_cfg.update(any_copy(npuw_llm_props));

    // m_cfg should be updated before checking for optimize_fp8, because affect the decision on kv-cache storage type
    auto kv_kache_storage_type = choose_kv_cache_storage_type(model, m_cfg, other_props);

    // Solely used for serialization at the moment
    m_non_llm_props = other_props;

    refine_dynamic_props(npuw_llm_props, npudesc);
    m_cfg.update(any_copy(npuw_llm_props));

    // Decide on using fused flash attention tile based on provided option and NPU capabilities.
    // If hardware supports and attention hint is set to HFA, then we can use fused flash attention implementation
    // automatically, unless user explicitly disables it via NPUW_ATTN_HFA_FUSED=NO option.
    const auto is_hfa =
        m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT>() == ::intel_npu::npuw::llm::AttentionHint::HFA;
    const auto hfa_fused_npu_supported = npudesc.has_value() && npudesc->support_flash_attention_tile;
    if (other_props.count("NPUW_ATTN_HFA_FUSED") == 0 && is_hfa && hfa_fused_npu_supported) {
        other_props["NPUW_ATTN_HFA_FUSED"] = "YES";
        LOG_INFO("Set NPUW_ATTN_HFA_FUSED to YES");
    }

    m_is_whisper = m_cfg.get<::intel_npu::NPUW_WHISPER>();
    if (m_is_whisper) {
        m_cfg.update({{"NPUW_LLM_SHARED_HEAD", "NO"}});
        m_cfg.update({{"NPUW_LLM_PREFILL_CHUNK_SIZE", "0"}});
        m_cfg.update({{"NPUW_LLM_CACHE_ROPE", "NO"}});
        m_cfg.update({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}});

        m_eos_token_id = m_cfg.get<::intel_npu::NPUW_WHISPER_EOS_TOKEN>();
    }

    m_is_eagle = use_eagle_key.value_or(false).as<bool>() == true;
    if (m_is_eagle) {
        LOG_INFO("Eagle3 speculative decoding mode enabled");
    }

    // Auto-detect MoE model by scanning for router/expert nodes
    const bool is_moe = is_moe_model(model);
    if (is_moe) {
        // Only apply MoE defaults if not explicitly set in external config
        if (npuw_llm_props.find("NPUW_LLM_SHARED_HEAD") == npuw_llm_props.end()) {
            m_cfg.update({{"NPUW_LLM_SHARED_HEAD", "NO"}});
        }
        if (npuw_llm_props.find("NPUW_LLM_GENERATE_HINT") == npuw_llm_props.end()) {
            m_cfg.update({{"NPUW_LLM_GENERATE_HINT", "BEST_PERF"}});
        }

        // Enable DEVICE_ROUTED mode by default for MoE models on newer compiler versions, as it's more efficient than
        // HOST_ROUTED
        if (npuw_llm_props.find("NPUW_LLM_GENERATE_MOE_HINT") == npuw_llm_props.end() && npudesc->arch == "5010" &&
            npudesc->compiler_ver >= ONEAPI_MAKE_VERSION(7, 29)) {
            m_cfg.update({{"NPUW_LLM_GENERATE_MOE_HINT", "DEVICE_ROUTED"}});
        }
    }

    // NB: PREFILL_HINT is now compatible with the PREFILL_CONFIG section, unlike for
    // the generate model they're not mutually exclusive
    const ::intel_npu::npuw::llm::PrefillHint prefill_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_HINT>();
    m_prefill_chunk_size = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_CHUNK_SIZE>();
    m_use_chunk_prefill = (prefill_hint == ::intel_npu::npuw::llm::PrefillHint::DYNAMIC && m_prefill_chunk_size > 0);

    uint32_t max_prompt_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t min_response_len = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);
    uint32_t max_generation_token_len = m_cfg.get<::intel_npu::NPUW_LLM_MAX_GENERATION_TOKEN_LEN>();
    if (max_generation_token_len != 1) {
        max_generation_token_len = align_to(max_generation_token_len, 8u);
    }

    // If chunk size covers the entire prompt, just follow the static behavior.
    // Otherwise, use chunking and align the prompt size to the chunk size.
    if (m_use_chunk_prefill) {
        OPENVINO_ASSERT(
            !ov::npuw::util::has_input(model, "token_type_ids") || !ov::npuw::util::has_input(model, "inputs_embeds"),
            "Chunking is not implemented for Gemma model family yet. "
            "Please set NPUW_LLM_PREFILL_HINT to 'STATIC'");
        if (m_prefill_chunk_size >= max_prompt_len) {
            m_use_chunk_prefill = false;
        } else {
            const auto is_power_of_two = [](uint64_t n) {
                return n > 0 && (n & (n - 1)) == 0;
            };
            if (!is_power_of_two(m_prefill_chunk_size)) {
                OPENVINO_THROW("Configuration Error: chunk size (",
                               m_prefill_chunk_size,
                               ") is not power of 2. Please adjust NPUW_LLM_PREFILL_CHUNK_SIZE.");
            }
            max_prompt_len = align_to(max_prompt_len, static_cast<uint32_t>(m_prefill_chunk_size));
        }

        m_enable_prefix_caching = m_cfg.get<::intel_npu::NPUW_LLM_ENABLE_PREFIX_CACHING>();
        if (m_enable_prefix_caching) {
            LOG_INFO("Prefix caching is enabled");
            m_prefix_caching_block_size = m_cfg.get<::intel_npu::NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE>();
            if (!is_aligned_to(static_cast<uint32_t>(m_prefill_chunk_size),
                               static_cast<uint32_t>(m_prefix_caching_block_size))) {
                LOG_INFO("Prefix caching block size is adjusted to " << m_prefill_chunk_size);
                m_prefix_caching_block_size = m_prefill_chunk_size;
            }
            m_prefix_caching_max_num_blocks = m_cfg.get<::intel_npu::NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS>();
            LOG_INFO("Prefix caching block size: " << m_prefix_caching_block_size);
            LOG_INFO("Prefix caching maximum number of blocks: " << m_prefix_caching_max_num_blocks);
        }
    }

    LOG_VERB("Enabled prefill chunking: " << m_use_chunk_prefill);
    LOG_VERB("Prefill chunk size: " << m_prefill_chunk_size);
    LOG_VERB("Maximum prompt length: " << max_prompt_len);

    const uint32_t batch_dim = m_cfg.get<::intel_npu::NPUW_LLM_BATCH_DIM>();
    const uint32_t seq_len_dim = m_cfg.get<::intel_npu::NPUW_LLM_SEQ_LEN_DIM>();
    KVAxesPosition axes{batch_dim, seq_len_dim};

    LOG_DEBUG("Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();

    auto use_text_embed_key = pop_option(other_props, std::string("NPUW_TEXT_EMBED"));
    m_is_embedding = use_text_embed_key.value_or(false).as<bool>() == true;

    if (m_is_embedding) {
        LOG_DEBUG("Text-embedding model rebuild");
        ov::npuw::util::PrepareTextEmbeddingModel(seq_len_dim).run_on_model(kvcache_model);
    } else {
        LOG_DEBUG("Transform kvcache model from stateful to stateless.");
        ov::pass::StatefulToStateless().run_on_model(kvcache_model);
    }

    ov::npuw::LoraStatefulToStatelessPass().run_on_model(kvcache_model);

    LOG_DEBUG("   ...also convert BF16 to FP16");
    // Note: we need to identify original bf16 constants for potential weightless deserialization later
    // And only then do bf16 to f16 transformation
    m_bf16_consts = ov::npuw::s11n::get_bf16_consts(model);
    ov::pass::ConvertPrecision(ov::element::bf16, ov::element::f16).run_on_model(kvcache_model);

    auto lm_head_model = check_and_cut_lm_head(kvcache_model, m_cfg);

    if (!m_is_whisper) {
        LOG_DEBUG("Try patch Phi-3 sliding window mask, if it exists.");
        ov::npuw::PatchPhi3SlidingMask().run_on_model(kvcache_model);
    }

    LOG_DEBUG("Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");

    m_kvcache_desc =
        KVCacheDesc{max_prompt_len, max_prompt_len + min_response_len, 0u, seq_len_dim, max_generation_token_len};

    uint32_t whisper_lhs_seq_size = 0;  // Not applicable for LLMs/VLMs
    if (m_is_whisper) {
        axes = KVAxesPosition{whisper_batch_dim, whisper_seq_len_dim};
        m_kvcache_desc = KVCacheDesc{whisper_max_prompt_size, whisper_kvcache_size, 0u, whisper_seq_len_dim, 1u};
        whisper_lhs_seq_size =
            static_cast<uint32_t>(prefill_model->input("encoder_hidden_states").get_partial_shape()[1].get_length());
        auto whisper_decompose_sdpa = m_cfg.get<::intel_npu::NPUW_WHISPER_DECOMPOSE_SDPA>();
        if (whisper_decompose_sdpa) {
            m_kvcache_desc.max_prompt_size = whisper_kvcache_size - 1;
        }

        auto prepare_prefill_model = ov::npuw::util::PrepareWhisperPrefillModel(m_kvcache_desc.max_prompt_size,
                                                                                whisper_lhs_seq_size,
                                                                                whisper_decompose_sdpa);
        prepare_prefill_model.run_on_model(prefill_model);                         // Whisper decoder model
        ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(kvcache_model);  // Whisper decoder_with_past model

        // FIXME: Whisper Decompose SDPA
        // WA: to mock new "cross_attention_qk_scaled_scores" outputs in original model
        if (whisper_decompose_sdpa) {
            m_decomposed_sdpa_size = prepare_prefill_model.get_decomposed_sdpa_size();
            auto& mutable_outputs = const_cast<std::vector<ov::Output<const ov::Node>>&>(this->outputs());
            for (size_t idx = 0; idx < m_decomposed_sdpa_size; idx++) {
                auto fake_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
                auto fake_result = std::make_shared<ov::op::v0::Result>(fake_param);
                fake_result->output(0).get_tensor().add_names(
                    {"cross_attention_qk_scaled_scores", "cross_attention_qk_scaled_scores_" + std::to_string(idx)});

                mutable_outputs.emplace_back(fake_result->output(0));
            }
        }
    }

    LOG_DEBUG("Make prefill model with static shapes");
    m_max_lora_rank = m_cfg.get<::intel_npu::NPUW_LLM_MAX_LORA_RANK>();
    if (m_use_chunk_prefill) {
        ov::npuw::ReshapeToStatic(static_cast<uint32_t>(m_prefill_chunk_size),
                                  m_kvcache_desc.max_prompt_size,
                                  axes,
                                  m_max_lora_rank,
                                  0,
                                  true)
            .run_on_model(prefill_model);
    } else {
        ov::npuw::ReshapeToStatic(m_kvcache_desc.max_prompt_size,
                                  m_kvcache_desc.max_prompt_size,
                                  axes,
                                  m_max_lora_rank,
                                  whisper_lhs_seq_size,
                                  true)
            .run_on_model(prefill_model);
    }
    LOG_DEBUG("Make kvcache model with static shapes");

    // Create generate model variants with different sizes
    auto generate_model_variants = create_generate_model_variants(kvcache_model, axes, whisper_lhs_seq_size);

    if (lm_head_model) {
        LOG_DEBUG("Shared LM head: slice the prefill output");
        // KVCache model is already reshaped to [1, max_generation_token_len, embed size],
        // so only apply slice to the Prefill model:
        ov::npuw::SliceOutEmbeds(axes.batch, m_kvcache_desc.max_generation_token_len).run_on_model(prefill_model);
        LOG_DEBUG("Make LM head model with static shapes");
        ov::npuw::ReshapeSlicedHeadToStatic(axes.batch, m_kvcache_desc.max_generation_token_len)
            .run_on_model(lm_head_model);
    }

    LOG_DEBUG("5.1, decompose GroupQueryAttention OP");
    ov::npuw::DecomposeGQA(true).run_on_model(prefill_model);
    for (auto& model_variant : generate_model_variants) {
        ov::npuw::DecomposeGQA(false).run_on_model(model_variant);
    }

    const auto prefill_attn_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT>();
    const auto generate_attn_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_ATTENTION_HINT>();
    const bool prefill_attn_dyn = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::DYNAMIC;
    const bool generate_attn_dyn = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::DYNAMIC;

    const bool prefill_attn_pyramid = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::PYRAMID;
    const bool generate_attn_pyramid = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::PYRAMID;

    const bool prefill_attn_hfa = prefill_attn_hint == ::intel_npu::npuw::llm::AttentionHint::HFA;
    const bool generate_attn_hfa = generate_attn_hint == ::intel_npu::npuw::llm::AttentionHint::HFA;

    const bool optimize_v_tensors = m_cfg.get<::intel_npu::NPUW_LLM_OPTIMIZE_V_TENSORS>();
    if (optimize_v_tensors) {
        LOG_DEBUG("Check and apply opt layout");
        LOG_BLOCK();
        // Only optimize V tensors for static attention types
        if (!generate_attn_dyn) {
            // Apply optimization to all variants and track results
            size_t optimized_count = 0;
            for (auto& model_variant : generate_model_variants) {
                if (ov::npuw::util::OptimizeValueTensors(false).run_on_model(model_variant)) {
                    ++optimized_count;
                }
            }

            // Check consistency: either all or none should be optimized
            if (optimized_count == generate_model_variants.size()) {
                LOG_DEBUG("V-tensors transposed in generate model variants");
                m_kvcache_desc.v_tensors_transposed_gen = true;
            } else if (optimized_count == 0) {
                LOG_DEBUG("No V-tensors were optimized");
                m_kvcache_desc.v_tensors_transposed_gen = false;
            } else {
                // Partial optimization is not allowed
                OPENVINO_ASSERT(false,
                                "Partial optimization detected: ",
                                optimized_count,
                                " out of ",
                                generate_model_variants.size(),
                                " variants were optimized, which is not allowed.");
            }
        }
        if (!prefill_attn_dyn && ov::npuw::util::OptimizeValueTensors(true).run_on_model(prefill_model)) {
            LOG_DEBUG("V-tensors tranposed in prefill model");
            m_kvcache_desc.v_tensors_transposed_pre = true;
        }
    } else {
        LOG_DEBUG("Check and apply opt layout --- SKIPPED");
    }
    if (!m_is_embedding) {
        if (!m_use_chunk_prefill) {
            LOG_DEBUG("Removing EmptyKVInputs");
            NPUW_ASSERT(ov::npuw::RemoveEmptyKVInputs().run_on_model(prefill_model));
        } else {
            LOG_DEBUG("Don't remove input key/values from prefill model.");
            LOG_DEBUG("Ask prefill model to output key/values for prefill chunk size tokens.");
            ov::npuw::RedirectNewKvToOutput().run_on_model(prefill_model);
        }

        LOG_DEBUG("Optimize generate model to output key/values for new token.");
        for (size_t i = 0; i < generate_model_variants.size(); ++i) {
            ov::npuw::RedirectNewKvToOutput().run_on_model(generate_model_variants[i]);
        }
    }
    LOG_DEBUG("Converting KV-cache in generate model to" << kv_kache_storage_type);
    for (size_t i = 0; i < generate_model_variants.size(); ++i) {
        ov::npuw::ConvertKVCacheToPrecision(kv_kache_storage_type).run_on_model(generate_model_variants[i]);
    }
    LOG_DEBUG("Converting KV-cache in prefill model to" << kv_kache_storage_type);
    ov::npuw::ConvertKVCacheToPrecision(kv_kache_storage_type).run_on_model(prefill_model);

    auto prefill_config =
        prefill_config_opt.value_or(get_default_prefill_config(prefill_model, npudesc)).as<ov::AnyMap>();

    // NB: GENERATE_HINT is only applicable for default generate config!
    if (generate_config_opt.has_value() && npuw_llm_props.count(ov::intel_npu::npuw::llm::generate_hint.name())) {
        OPENVINO_THROW("GENERATE_HINT only works with default generate config!");
    }
    const ::intel_npu::npuw::llm::GenerateHint generate_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_HINT>();
    auto generate_config =
        generate_config_opt.value_or(get_default_generate_config(npudesc, generate_hint)).as<ov::AnyMap>();

    auto prefill_config_addition_value =
        prefill_config_addition.has_value() ? prefill_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};
    auto generate_config_addition_value =
        generate_config_addition.has_value() ? generate_config_addition.value().as<ov::AnyMap>() : ov::AnyMap{};

    merge_config_with(prefill_config, other_props);
    merge_config_with(generate_config, other_props);
    merge_config_with(prefill_config, prefill_config_addition_value);
    merge_config_with(generate_config, generate_config_addition_value);

    // Convert LLM-specific attention hints to NPUW_ATTN
    if (npuw_llm_props.count("NPUW_LLM_PREFILL_ATTENTION_HINT")) {
        prefill_config["NPUW_ATTN"] = npuw_llm_props["NPUW_LLM_PREFILL_ATTENTION_HINT"];
    }
    if (npuw_llm_props.count("NPUW_LLM_GENERATE_ATTENTION_HINT")) {
        generate_config["NPUW_ATTN"] = npuw_llm_props["NPUW_LLM_GENERATE_ATTENTION_HINT"];
    }

    // Generate a random weights bank name unique to this LLMCompiledModel object
    auto weights_bank_name = ov::npuw::util::generate_random_string();
    LOG_VERB("Generated a unique weights bank name: " << weights_bank_name);
    apply_weights_bank_name(prefill_config, weights_bank_name);
    apply_weights_bank_name(generate_config, weights_bank_name);

    // Handle attention hints. FIXME: Maybe it makes sense to make those
    // mutually exclusive with the precise configuration sections as well
    const ov::AnyMap dyn_attn_opts = {
        {"NPUW_ONLINE_PIPELINE", "REP"},
        {"NPUW_ONLINE_ISOLATE", "ATTN"},
        {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
        {"NPUW_UNFOLD_IREQS", "NO"},
    };
    if (prefill_attn_dyn || prefill_attn_pyramid || prefill_attn_hfa) {
        merge_config_with(prefill_config, dyn_attn_opts);
    }
    if (generate_attn_dyn || generate_attn_pyramid || generate_attn_hfa) {
        merge_config_with(generate_config, dyn_attn_opts);
    }
    if (is_moe) {
        // Apply MoE configuration for prefill stage
        const auto prefill_moe_hint = m_cfg.get<::intel_npu::NPUW_LLM_PREFILL_MOE_HINT>();
        apply_moe_config(prefill_config, prefill_moe_hint, "PREFILL");

        // Apply MoE configuration for generate stage
        const auto generate_moe_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_MOE_HINT>();
        apply_moe_config(generate_config, generate_moe_hint, "GENERATE");

        // Apply model transformations only to GENERATE stage (PREFILL doesn't support DEVICE_ROUTED transformations)
        if (generate_moe_hint == ::intel_npu::npuw::llm::MoEHint::DEVICE_ROUTED) {
            LOG_INFO("Applying DEVICE_ROUTED MoE transformations to " << generate_model_variants.size() << " variants");
            for (auto&& model_variant : generate_model_variants) {
                ov::npuw::ApplyMoEDeviceRoutedTransforms().run_on_model(model_variant);
            }
            LOG_INFO("DEVICE_ROUTED MoE transformations completed");
        }
    }
    // Note: with dynamic attention in EITHER STAGE, we have to
    // explicitly disable the run-time fallback to so extra ov::Model
    // references won't be held by the npuw::CompiledModel, resulting
    // in a higher memory consumption. This behavior should be reworked!
    // The reason here is that NPUW_DEVICES may come as a global setting,
    // impacting all the stages.
    if (prefill_attn_dyn || generate_attn_dyn) {
        const ov::AnyMap no_runtime_fallback = {{"NPUW_FALLBACK_EXEC", "NO"}};
        merge_config_with(prefill_config, no_runtime_fallback);
        merge_config_with(generate_config, no_runtime_fallback);
    }

    if (m_is_whisper) {
        update_config_for_whisper(prefill_config);
        if (is_int8_compressed(model)) {
            disable_ws_for_whisper(prefill_config);
            disable_ws_for_whisper(generate_config);
            LOG_INFO(" WS is disabled for Whisper int8 model!");
        }
    }

    if (m_is_embedding) {
        update_config_for_text_embed(prefill_config);
    }

    if (m_cfg.get<::intel_npu::NPUW_LLM_CACHE_ROPE>()) {
        LOG_DEBUG("Caching preROPE ");
        const uint32_t CACHE_ROPE_START = 2048;
        const bool is_best = (generate_hint == ::intel_npu::npuw::llm::GenerateHint::BEST_PERF);

        if (!is_best || (max_prompt_len >= CACHE_ROPE_START)) {
            LOG_DEBUG("Enable RoPE Cache for prefill");
            ov::npuw::patterns::pre_compute::RopeCache rope_prefill_cacher(
                max_prompt_len,
                ov::npuw::LLMInferRequest::layer_names::longrope_input);
            rope_prefill_cacher.run_on_model(prefill_model);
        }

        // Apply RoPE Cache to all generate variant models
        for (size_t i = 0; i < generate_model_variants.size(); ++i) {
            const uint32_t kv_size = m_kvcache_sizes[i];
            if (!is_best || (kv_size >= CACHE_ROPE_START)) {
                LOG_DEBUG("Enable RoPE Cache for generate variant with size: " << kv_size);
                ov::npuw::patterns::pre_compute::RopeCache rope_cacher(
                    kv_size,
                    ov::npuw::LLMInferRequest::layer_names::longrope_input);
                rope_cacher.run_on_model(generate_model_variants[i]);
            }
        }
    }

    // Regularize models for the better partitioning assuming it is a transformer
    // Apply these transformations to all variant models
    {
        ov::npuw::patterns::regularize::RegularizeSDPA(prefill_attn_dyn || prefill_attn_pyramid || prefill_attn_hfa)
            .run_on_model(prefill_model);
        for (auto& model_variant : generate_model_variants) {
            ov::npuw::patterns::regularize::RegularizeSDPA(generate_attn_dyn || generate_attn_pyramid ||
                                                           generate_attn_hfa)
                .run_on_model(model_variant);
        }
    }

    // Compile multiple generate model variants with different sizes
    compile_generate_model_variants(generate_model_variants, plugin, generate_config);

    m_prefill_compiled = m_compiled_model_factory(prefill_model, plugin, prefill_config);
    NPUW_ASSERT(m_prefill_compiled && "Can't create ov::npuw::CompiledModel for passed prefill "
                                      "model and its config, please check passed config.");
    if (lm_head_model) {
        auto lm_head_config = get_default_lm_head_config(npudesc);
        merge_config_with(lm_head_config, other_props);
        auto lm_head_config_addition_value = lm_head_config_addition.value_or(ov::AnyMap{}).as<ov::AnyMap>();
        merge_config_with(lm_head_config, lm_head_config_addition_value);

        apply_weights_bank_name(lm_head_config, weights_bank_name);

        m_lm_head_compiled = m_compiled_model_factory(lm_head_model, plugin, lm_head_config);
        NPUW_ASSERT(m_lm_head_compiled);
    }

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
    // Serilize LLMCompiledModel identifier
    write(stream, NPUW_LLM_COMPILED_MODEL_INDICATOR);
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
        CompiledContext ctx(false, nullptr, nullptr);
        return serialize(stream, ctx);
    }

    // In case of weightless flow the whole blob will be encrypted on NPUW side.
    std::stringstream non_encrypted_stream;
    if (is_weightless) {
        non_encrypted_stream.copyfmt(stream);
        CompiledContext ctx(false, nullptr, nullptr);
        serialize(non_encrypted_stream, ctx);
        std::string encrypted = enc_callbacks.encrypt(non_encrypted_stream.str());
        write(stream, encrypted);
    } else {
        // In case of blob with weights only encrypt XML part of the model
        CompiledContext ctx(true, enc_callbacks.encrypt, nullptr);
        serialize(stream, ctx);
    }
}

void ov::npuw::LLMCompiledModel::serialize(std::ostream& raw_stream, const ov::npuw::s11n::CompiledContext& ctx) const {
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
        auto stream = Stream::writer(model_stream);
        // Serialize name
        stream & m_name;

        // Serialize inputs and outputs
        stream& inputs() & outputs();

        // Serialize LLMCompiledModel-specific data
        stream & m_kvcache_desc.max_prompt_size & m_kvcache_desc.total_size & m_kvcache_desc.num_stored_tokens &
            m_kvcache_desc.dim & m_kvcache_desc.max_generation_token_len & m_kvcache_desc.v_tensors_transposed_pre &
            m_kvcache_desc.v_tensors_transposed_gen & m_prefill_chunk_size & m_use_chunk_prefill & m_max_lora_rank &
            m_enable_prefix_caching & m_prefix_caching_block_size & m_prefix_caching_max_num_blocks & m_is_whisper &
            m_eos_token_id & m_is_eagle & m_is_embedding;

        // Write config
        stream & m_cfg;

        // Serialize KV cache model variants
        auto variant_count = static_cast<uint32_t>(m_generate_compiled_variants.size());
        stream & m_kvcache_sizes & variant_count;

        // Serialize CompiledModels
        // Note: no need to pass any encryption here as it's done in export_model()
        CompiledContext enc_ctx(false, nullptr, nullptr, m_bf16_consts);

        // Serialize all generate variants
        for (const auto& compiled_variant : m_generate_compiled_variants) {
            compiled_variant->serialize(model_stream, enc_ctx);
        }

        m_prefill_compiled->serialize(model_stream, enc_ctx);
        const bool is_shared_lm_head = m_lm_head_compiled != nullptr;
        stream & is_shared_lm_head;
        if (is_shared_lm_head) {
            m_lm_head_compiled->serialize(model_stream, enc_ctx);
        }
    };

    std::stringstream non_encrypted_stream;
    if (ctx.encrypted) {
        NPUW_ASSERT(ctx.encrypt && "Encryption function isn't provided!");
        non_encrypted_stream.copyfmt(raw_stream);
        write_model_meta(non_encrypted_stream);
        std::string encrypted_str = ctx.encrypt(non_encrypted_stream.str());
        write(raw_stream, encrypted_str);
    } else {
        write_model_meta(raw_stream);
    }

    // Serialize bank name
    const auto& kv_bank = m_kvcache_compiled->get_weights_bank();
    const auto& p_bank = m_prefill_compiled->get_weights_bank();
    NPUW_ASSERT(kv_bank && p_bank && kv_bank == p_bank && "Prefill and KVCache models' weight bank should be shared!");
    auto stream = Stream::writer(raw_stream);
    auto bank_name = kv_bank->get_name();
    stream & bank_name;

    if (!is_weightless) {
        ov::npuw::s11n::serialize(stream, *kv_bank);
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

    ov::npuw::s11n::IndicatorType llm_compiled_indicator;
    read(stream, llm_compiled_indicator);
    NPUW_ASSERT(llm_compiled_indicator == NPUW_LLM_COMPILED_MODEL_INDICATOR &&
                "This blob wasn't serialized via LLMCompiledModel!");

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
        auto stream = Stream::reader(model_stream);
        std::string bank_name;
        stream & bank_name;

        if (is_weightless) {
            auto bank = ov::npuw::weights::bank(bank_name, compiled->get_plugin()->get_core(), "");

            for (const auto& compiled_variant : compiled->m_generate_compiled_variants) {
                compiled_variant->set_weights_bank(bank);
                compiled_variant->finalize_weights_bank();
            }

            compiled->m_prefill_compiled->set_weights_bank(bank);
            compiled->m_prefill_compiled->finalize_weights_bank();

            if (compiled->m_lm_head_compiled) {
                compiled->m_lm_head_compiled->set_weights_bank(bank);
                compiled->m_lm_head_compiled->finalize_weights_bank();
            }
        } else {
            auto bank = ov::npuw::weights::bank(bank_name, compiled->get_plugin()->get_core(), "");
            ov::npuw::s11n::serialize(stream, *bank);

            compiled->m_kvcache_compiled->set_weights_bank(bank);
            for (const auto& compiled_variant : compiled->m_generate_compiled_variants) {
                compiled_variant->set_weights_bank(bank);
                compiled_variant->reconstruct_closure();
            }

            compiled->m_prefill_compiled->set_weights_bank(bank);
            compiled->m_prefill_compiled->reconstruct_closure();

            if (compiled->m_lm_head_compiled) {
                compiled->m_lm_head_compiled->set_weights_bank(bank);
                compiled->m_lm_head_compiled->reconstruct_closure();
            }
        }
    };

    if (!encrypted) {
        CompiledContext ctx(false, nullptr, nullptr);
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
        CompiledContext ctx(false, nullptr, nullptr);
        compiled_model = ov::npuw::LLMCompiledModel::deserialize(decrypted_stream, plugin, properties, ctx);
    } else {
        CompiledContext ctx(true, nullptr, enc_callbacks.decrypt);
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
    const ov::npuw::s11n::CompiledContext& ctx) {
    using namespace ov::npuw::s11n;

    auto read_model_meta = [&](std::istream& model_stream) {
        auto stream = Stream::reader(model_stream);
        // Deserialize model name first
        std::string model_name;
        stream & model_name;

        // Create a dummy CompiledModel with an empty ov::Model - this will skip the constructor flow
        // to continue deserialization
        ov::ParameterVector parameters;
        ov::NodeVector results;

        stream & parameters & results;

        auto ov_model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters, model_name);

        auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(ov_model, plugin, true);

        // Deserialize LLMCompiledModel-specific data
        stream & compiled->m_kvcache_desc.max_prompt_size & compiled->m_kvcache_desc.total_size &
            compiled->m_kvcache_desc.num_stored_tokens & compiled->m_kvcache_desc.dim &
            compiled->m_kvcache_desc.max_generation_token_len & compiled->m_kvcache_desc.v_tensors_transposed_pre &
            compiled->m_kvcache_desc.v_tensors_transposed_gen & compiled->m_prefill_chunk_size &
            compiled->m_use_chunk_prefill & compiled->m_max_lora_rank & compiled->m_enable_prefix_caching &
            compiled->m_prefix_caching_block_size & compiled->m_prefix_caching_max_num_blocks & compiled->m_is_whisper &
            compiled->m_eos_token_id & compiled->m_is_eagle & compiled->m_is_embedding;

        // Deserialize config
        stream & compiled->m_cfg;
        compiled->implement_properties();

        // Deserialize KV cache model variants
        stream & compiled->m_kvcache_sizes;
        uint32_t num_variants = 0;
        stream & num_variants;

        compiled->m_generate_compiled_variants.reserve(num_variants);

        // Deserialize CompiledModels
        // Note: no need to pass any encryption here as it's done in import_model()
        CompiledContext enc_ctx(false, nullptr, nullptr);

        // Deserialize all generate variants
        for (uint32_t i = 0; i < num_variants; ++i) {
            auto compiled_variant = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
            compiled->m_generate_compiled_variants.push_back(compiled_variant);
        }

        // Set the main kvcache_compiled to the largest variant for backward compatibility
        if (!compiled->m_generate_compiled_variants.empty()) {
            compiled->m_kvcache_compiled = compiled->m_generate_compiled_variants.back();
        }

        compiled->m_prefill_compiled = ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
        bool is_shared_lm_head = false;
        stream & is_shared_lm_head;
        if (is_shared_lm_head) {
            compiled->m_lm_head_compiled =
                ov::npuw::CompiledModel::deserialize(model_stream, plugin, properties, enc_ctx);
        }

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
    if (m_is_whisper) {
        return non_const_this->create_whisper_infer_request();
    } else if (m_is_embedding) {
        return non_const_this->create_embedding_infer_request();
    } else {
        return non_const_this->create_llm_infer_request();
    }
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_llm_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_whisper_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::WhisperInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::LLMCompiledModel::create_embedding_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::LLMCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::EmbeddingInferRequest>(this_sptr);
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
                          BIND(npuw::llm::seq_len_dim, NPUW_LLM_SEQ_LEN_DIM, get),
                          BIND(npuw::llm::max_prompt_len, NPUW_LLM_MAX_PROMPT_LEN, get),
                          BIND(npuw::llm::min_response_len, NPUW_LLM_MIN_RESPONSE_LEN, get),
                          BIND(npuw::llm::optimize_v_tensors, NPUW_LLM_OPTIMIZE_V_TENSORS, get),
                          BIND(npuw::llm::optimize_fp8, NPUW_LLM_OPTIMIZE_FP8, get),
                          BIND(npuw::llm::cache_rope, NPUW_LLM_CACHE_ROPE, get),
                          BIND(npuw::llm::prefill_moe_hint, NPUW_LLM_PREFILL_MOE_HINT, get),
                          BIND(npuw::llm::generate_moe_hint, NPUW_LLM_GENERATE_MOE_HINT, get),
                          BIND(npuw::llm::generate_pyramid, NPUW_LLM_GENERATE_PYRAMID, get),
                          BIND(npuw::llm::prefill_chunk_size, NPUW_LLM_PREFILL_CHUNK_SIZE, get),
                          BIND(npuw::llm::prefill_hint, NPUW_LLM_PREFILL_HINT, getString),
                          BIND(npuw::llm::generate_hint, NPUW_LLM_GENERATE_HINT, getString),
                          BIND(npuw::llm::prefill_attn_hint, NPUW_LLM_PREFILL_ATTENTION_HINT, getString),
                          BIND(npuw::llm::generate_attn_hint, NPUW_LLM_GENERATE_ATTENTION_HINT, getString),
                          BIND(npuw::llm::shared_lm_head, NPUW_LLM_SHARED_HEAD, get),
                          BIND(npuw::whisper::enabled, NPUW_WHISPER, get),
                          BIND(npuw::whisper::whisper_eos_token, NPUW_WHISPER_EOS_TOKEN, get),
                          BIND(npuw::whisper::whisper_decompose_sdpa, NPUW_WHISPER_DECOMPOSE_SDPA, get),
                          BIND(npuw::eagle::enabled, NPUW_EAGLE, get),
                          BIND(npuw::text_embed::enabled, NPUW_TEXT_EMBED, get)});
#undef BIND
}
