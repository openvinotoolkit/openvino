// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "llm_compiled_model.hpp"

#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace {
uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
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

template <typename T>
std::optional<T> get_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        return std::make_optional(it->second.as<T>());
    }
    return std::nullopt;
}

template <typename T>
T pop_or_default(ov::AnyMap& config, const std::string& key, const T& default_value) {
    auto anyopt = pop_option(config, key);
    if (anyopt.has_value()) {
        return anyopt.value().as<T>();
    }
    return default_value;
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

void drop_cache_dir(ov::AnyMap& config) {
    if (config.count("NPU_USE_NPUW") != 0u) {
        pop_option(config, "CACHE_DIR");
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
    m_cfg.update(any_copy(npuw_llm_props));

    LOG_DEBUG("1. Creating kvcache model as clone of passed one.");
    auto kvcache_model = model->clone();
    LOG_DEBUG("2. Transform kvcache model from stateful to stateless.");
    ov::pass::StatefulToStateless().run_on_model(kvcache_model);

    LOG_DEBUG("3. Creating prefill model as clone of transformed kvcache one.");
    auto prefill_model = kvcache_model->clone();
    prefill_model->set_friendly_name(kvcache_model->get_friendly_name() + "_prefill");
    LOG_DEBUG("4. Converting KV-cache in prefill model to FP16.");
    prefill_model = cvt_kvcache_to_fp16(prefill_model);

    LOG_DEBUG("5. Optimize kvcache kvcache model to output key/values for new token.");
    kvcache_model = redirect_new_kv_to_output(kvcache_model);
    LOG_DEBUG("6. Converting KV-cache in kvcache model to FP16.");
    kvcache_model = cvt_kvcache_to_fp16(kvcache_model);

    const uint32_t kMaxPromptLen = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>(), 64u);
    const uint32_t kMinResponseLen = align_to(m_cfg.get<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>(), 64u);
    const ::intel_npu::npuw::llm::ModelDesc model_desc = m_cfg.get<::intel_npu::NPUW_LLM_MODEL_DESC>();
    KVAxesPosition axes = get_kv_axes(model_desc.type);
    m_kvcache_desc = KVCacheDesc{kMaxPromptLen, kMaxPromptLen + kMinResponseLen, 0u, axes.seq_len};
    LOG_DEBUG("7. Make prefill model with static shapes");
    reshape_to_static(prefill_model, m_kvcache_desc.max_prompt_size, m_kvcache_desc.max_prompt_size, axes);
    LOG_DEBUG("8. Make kvcache model with static shapes");
    reshape_to_static(kvcache_model, 1u, m_kvcache_desc.total_size, axes);

    auto npudesc = extract_npu_descriptor(plugin);

    ov::AnyMap properties_copy = std::move(other_props);
    auto prefill_config = get_default_prefill_config(model, npudesc);
    // NB: GENERATE_HINT is only applicable for default generate config!
    const ::intel_npu::npuw::llm::GenerateHint generate_hint = m_cfg.get<::intel_npu::NPUW_LLM_GENERATE_HINT>();
    LOG_DEBUG("9. Passed GENERATE_HINT: " << std::string(::intel_npu::NPUW_LLM_GENERATE_HINT::toString(generate_hint)));
    auto generate_config = get_default_generate_config(model, npudesc, generate_hint);
    merge_config_with(prefill_config, properties_copy);
    merge_config_with(generate_config, properties_copy);
    // FIXME: Drop CACHE_DIR option if NPUW is enabled
    drop_cache_dir(prefill_config);
    drop_cache_dir(generate_config);

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
    return std::make_shared<ov::npuw::LLMInferRequest>(this_sptr, m_kvcache_desc);
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
