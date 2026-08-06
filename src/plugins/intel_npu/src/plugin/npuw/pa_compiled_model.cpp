// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

bool is_kv_cache_name(const std::string& name) {
    return ov::npuw::util::starts_with(name, "key_cache.") || ov::npuw::util::starts_with(name, "value_cache.");
}

// The PA control tensors are small i32/i64 vectors -- widen to i64 for checks.
std::vector<int64_t> as_i64_vec(const ov::SoPtr<ov::ITensor>& tensor) {
    const auto n = tensor->get_size();
    std::vector<int64_t> out(n);
    if (tensor->get_element_type() == ov::element::i32) {
        const auto* data = tensor->data<int32_t>();
        std::copy_n(data, n, out.begin());
    } else if (tensor->get_element_type() == ov::element::i64) {
        const auto* data = tensor->data<int64_t>();
        std::copy_n(data, n, out.begin());
    } else {
        OPENVINO_THROW("PA: unexpected element type ", tensor->get_element_type(), " for a control tensor");
    }
    return out;
}

// True when the model matches the plain flat-token LLM contract the chunked
// path implements: the known control inputs only (embedding inputs, M-RoPE
// position_ids, per-layer block tables etc. all run 1:1 on the dynamic model),
// 1-D token streams, and a single logits output with static per-row geometry
// so the result tensor can be allocated upfront and filled row by row.
bool is_chunkable_pa_model(const std::shared_ptr<ov::Model>& model) {
    static const std::unordered_set<std::string> known = {"input_ids",
                                                          "position_ids",
                                                          "past_lens",
                                                          "subsequence_begins",
                                                          "block_indices",
                                                          "block_indices_begins",
                                                          "max_context_len",
                                                          "score_aggregation_window",
                                                          "sampled_tokens_indices"};
    std::unordered_set<std::string> seen;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        if (is_kv_cache_name(name)) {
            continue;
        }
        if (known.count(name) == 0) {
            return false;
        }
        seen.insert(name);
    }
    for (const char* required : {"input_ids", "position_ids", "block_indices", "sampled_tokens_indices"}) {
        if (seen.count(required) == 0) {
            return false;
        }
    }
    for (const char* name : {"input_ids", "position_ids"}) {
        const auto& rank = model->input(name).get_partial_shape().rank();
        if (rank.is_dynamic() || rank.get_length() != 1) {
            return false;
        }
    }
    const auto& outputs = model->outputs();
    if (outputs.size() != 1 || outputs.front().get_any_name() != "logits") {
        return false;
    }
    const auto& lshape = outputs.front().get_partial_shape();
    return lshape.rank().is_static() && lshape.rank().get_length() == 3 && lshape[1].is_static() &&
           lshape[2].is_static();
}

std::shared_ptr<ov::Model> derive_pa_semi_static_model(const std::shared_ptr<ov::Model>& base_model,
                                                       std::size_t token_dim) {
    // Only the token-driven inputs get a fixed size (both are 1-D, checked by
    // is_chunkable_pa_model); the context stays dynamic.
    auto derived = base_model->clone();
    derived->reshape({{"input_ids", ov::PartialShape{static_cast<int64_t>(token_dim)}},
                      {"position_ids", ov::PartialShape{static_cast<int64_t>(token_dim)}}});
    derived->set_friendly_name(base_model->get_friendly_name() + "_pa_token_" + std::to_string(token_dim));
    return derived;
}

std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> compile_pa_semi_static_variants(
    const std::shared_ptr<ov::Model>& base_model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const std::string& device,
    const ov::AnyMap& inner_config) {
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> variants;
    constexpr std::array<std::size_t, 3> kVariantTokenDims = {1024u, 128u, 1u};

    for (const auto token_dim : kVariantTokenDims) {
        auto derived = derive_pa_semi_static_model(base_model, token_dim);
        auto compiled = plugin->get_core()->compile_model(derived, device, inner_config);
        OPENVINO_ASSERT(compiled != nullptr,
                        "PA semi-static derivation failed to compile token_dim=",
                        token_dim,
                        " on ",
                        device);
        LOG_INFO("PA: compiled semi-static variant token_dim=" << token_dim << " on " << device);
        variants.emplace(token_dim, std::move(compiled));
    }

    return variants;
}

}  // anonymous namespace

ov::npuw::PACompiledModel::PreparedState ov::npuw::PACompiledModel::prepare(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    // The fallback device is an internal development knob, not a config
    // option - an env var keeps it out of user configs (and blob cache keys).
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    // NPU cannot take the PA op itself (dynamic shapes, no PA kernel).
    OPENVINO_ASSERT(!ov::npuw::util::starts_with(device, "NPU"),
                    "OPENVINO_NPUW_PA_DEVICE must be the PagedAttention fallback device (CPU or GPU), got ",
                    device);

    LOG_INFO("PA: compiling the dynamic PA model 1:1 on " << device);

    // Sanity: this must be the model the CB pipeline deploys -- PA control
    // inputs plus a paged KV cache.
    bool has_past_lens = false, has_cache = false;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        has_past_lens |= (name == "past_lens");
        has_cache |= is_kv_cache_name(name);
    }
    OPENVINO_ASSERT(has_past_lens && has_cache,
                    "PACompiledModel expects the continuous-batching PA model "
                    "(past_lens + key_cache/value_cache inputs)");

    // The 1:1 part: the model is compiled exactly as received. NPUW_*,
    // NPU_USE_NPUW and NPU_* keys are this plugin's configuration and must not
    // reach the executing device (which would reject them as unsupported);
    // everything else (e.g. KV_CACHE_PRECISION, performance hints) is the
    // executing device's business and is forwarded.
    ov::AnyMap inner_config;
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            continue;
        }
        inner_config.emplace(key, value);
    }
    auto compiled = plugin->get_core()->compile_model(model, device, inner_config);
    OPENVINO_ASSERT(compiled != nullptr, "PACompiledModel requires a valid inner compiled model");

    // Stamp the device-resolved KV cache element types and shapes back onto
    // the model's cache Parameters. The source PA model declares them fully
    // dynamic (even the element type); the CB pipeline's KVCacheManager reads
    // cache precision and block geometry from *this* compiled model's ports,
    // so they must expose what the device actually decided.
    std::unordered_map<std::string, ov::Output<const ov::Node>> inner_inputs;
    for (const auto& input : compiled->inputs()) {
        inner_inputs.emplace(input.get_any_name(), input);
    }
    for (const auto& param : model->get_parameters()) {
        const auto& name = param->get_output_tensor(0).get_any_name();
        if (!is_kv_cache_name(name)) {
            continue;
        }
        auto it = inner_inputs.find(name);
        OPENVINO_ASSERT(it != inner_inputs.end(), "PA: inner compiled model lost the '", name, "' input");
        param->set_element_type(it->second.get_element_type());
        param->set_partial_shape(it->second.get_partial_shape());
    }
    model->validate_nodes_and_infer_types();

    // The semi-static variants only make sense for the plain flat-token LLM
    // contract; anything else (VLM, M-RoPE, per-layer block tables, ...) runs
    // 1:1 on the dynamic model, so don't spend compile time on variants.
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> semi_static_compiled;
    if (is_chunkable_pa_model(model)) {
        semi_static_compiled = compile_pa_semi_static_variants(model, plugin, device, inner_config);
    } else {
        LOG_INFO("PA: model is outside the chunkable flat-token contract; every dispatch runs 1:1");
    }
    return PreparedState{model, std::move(compiled), std::move(semi_static_compiled), std::move(device)};
}

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : PACompiledModel(prepare(model, plugin, properties), plugin) {}

ov::npuw::PACompiledModel::PACompiledModel(PreparedState prepared, const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_device(std::move(prepared.device)),
      m_compiled_model(std::move(prepared.compiled)),
      m_semi_static_models(std::move(prepared.semi_static_compiled)) {
    // The device fixes the KV cache geometry at compile time; remember the
    // block size for validating block-table coverage per dispatch.
    for (const auto& input : m_compiled_model->inputs()) {
        if (input.get_any_name() == "key_cache.0") {
            const auto& shape = input.get_partial_shape();
            // [num_blocks (dyn), kv_heads, block_size, head_size]
            if (shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static()) {
                m_block_size = static_cast<std::size_t>(shape[2].get_length());
            }
            break;
        }
    }
    LOG_INFO("PA: KV block_size fixed by " << m_device << ": " << m_block_size << "; " << m_semi_static_models.size()
                                           << " semi-static variant(s)");
}

void ov::npuw::PACompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not support export_model()");
}

std::shared_ptr<const ov::Model> ov::npuw::PACompiledModel::get_runtime_model() const {
    return m_compiled_model->get_runtime_model();
}

void ov::npuw::PACompiledModel::set_property(const ov::AnyMap& properties) {
    // The PA-level options are fixed at compile time; catching them here gives
    // a clear error instead of the executing device's "unsupported property".
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            OPENVINO_THROW("PACompiledModel: '", key, "' cannot be changed after the model is compiled");
        }
    }
    m_compiled_model->set_property(properties);
}

ov::Any ov::npuw::PACompiledModel::get_property(const std::string& name) const {
    // The PA-level key is answered here; everything else is the executing
    // device's business (notably ov::execution_devices, which the CB pipeline
    // queries to pick its block size).
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    if (name == ov::supported_properties.name()) {
        // Keep the property surface self-consistent: the inner device's list
        // plus the PA key answered above.
        auto props = m_compiled_model->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(std::string(::intel_npu::NPUW_PA::key()), ov::PropertyMutability::RO);
        return props;
    }
    return m_compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::PACompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const PACompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::PAInferRequest>(std::move(self));
}

ov::npuw::PAInferRequest::PAInferRequest(std::shared_ptr<const PACompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)) {
    m_inner_request = m_compiled_model->m_compiled_model->create_infer_request();
    OPENVINO_ASSERT(m_inner_request != nullptr, "PA infer request requires a valid inner request");
    for (const auto& input : m_inner_request->get_compiled_model()->inputs()) {
        m_inner_inputs.emplace(input.get_any_name(), input);
    }

    // Map outer ports to inner ports by tensor name, once. Matching by name
    // (not by position) keeps the forwarding correct even if the executing
    // device reorders ports, and gives O(1) lookups on the dispatch hot path.
    std::unordered_map<std::string, ov::Output<const ov::Node>> inner_outputs;
    for (const auto& output : m_inner_request->get_compiled_model()->outputs()) {
        inner_outputs.emplace(output.get_any_name(), output);
    }
    const auto map_ports = [this](const auto& outer_ports, const auto& inner_by_name) {
        for (const auto& outer : outer_ports) {
            auto it = inner_by_name.find(outer.get_any_name());
            OPENVINO_ASSERT(it != inner_by_name.end(),
                            "PA: inner compiled model has no port named '",
                            outer.get_any_name(),
                            "'");
            m_port_map.emplace(outer.get_node(), it->second);
        }
    };
    map_ports(m_compiled_model->inputs(), m_inner_inputs);
    map_ports(m_compiled_model->outputs(), inner_outputs);
}

const ov::Output<const ov::Node>& ov::npuw::PAInferRequest::map_port_locked(
    const ov::Output<const ov::Node>& port) const {
    auto it = m_port_map.find(port.get_node());
    OPENVINO_ASSERT(it != m_port_map.end(), "Unknown PA infer request port: ", port.get_any_name());
    return it->second;
}

ov::npuw::PAInferRequest::Dispatch ov::npuw::PAInferRequest::validate_dispatch_locked() {
    const auto get = [&](const char* name) {
        auto it = m_inner_inputs.find(name);
        OPENVINO_ASSERT(it != m_inner_inputs.end(), "PA model has no '", name, "' input");
        return m_inner_request->get_tensor(it->second);
    };

    const auto expect = [&](bool cond, const char* what) {
        OPENVINO_ASSERT(cond, "PA dispatch #", m_dispatch_idx, " violates the PA model expectations: ", what);
    };

    Dispatch d;
    d.past_lens = as_i64_vec(get("past_lens"));
    d.subsequence_begins = as_i64_vec(get("subsequence_begins"));
    const auto& past = d.past_lens;
    const auto& sub = d.subsequence_begins;
    const auto mcl_vec = as_i64_vec(get("max_context_len"));
    expect(!mcl_vec.empty(), "max_context_len is not set");
    const auto mcl = mcl_vec.front();
    d.n_seqs = static_cast<int64_t>(past.size());
    // subsequence_begins is the source of truth for the flat token dimension.
    // input_ids is absent on embedding-input models (inputs_embeds), so it is
    // only cross-checked when present; position_ids may be multi-dimensional
    // (M-RoPE), so its token count is the last shape dim.
    d.n_tokens = sub.empty() ? int64_t{0} : sub.back();

    if (m_inner_inputs.count("input_ids") > 0) {
        expect(static_cast<int64_t>(get("input_ids")->get_size()) == d.n_tokens,
               "input_ids size != subsequence_begins token count");
    }
    const auto& pos_shape = get("position_ids")->get_shape();
    expect(!pos_shape.empty() && static_cast<int64_t>(pos_shape.back()) == d.n_tokens,
           "position_ids last dim != subsequence_begins token count");
    expect(static_cast<int64_t>(sub.size()) == d.n_seqs + 1, "subsequence_begins size != past_lens size + 1");
    expect(sub.front() == 0, "subsequence_begins does not start at 0");
    expect(std::is_sorted(sub.begin(), sub.end()) && std::adjacent_find(sub.begin(), sub.end()) == sub.end(),
           "subsequence_begins is not strictly increasing");

    // The shared block table. Cache-eviction models carry per-layer
    // block_indices.<L> inputs instead; those dispatches run 1:1 and only the
    // common controls above are validated.
    const bool has_block_table = m_inner_inputs.count("block_indices") > 0;
    if (has_block_table) {
        d.block_indices = as_i64_vec(get("block_indices"));
        d.block_indices_begins = as_i64_vec(get("block_indices_begins"));
        const auto& bib = d.block_indices_begins;
        expect(static_cast<int64_t>(bib.size()) == d.n_seqs + 1, "block_indices_begins size != past_lens size + 1");
        expect(bib.front() == 0 && bib.back() == static_cast<int64_t>(d.block_indices.size()) &&
                   std::is_sorted(bib.begin(), bib.end()),
               "block_indices_begins is not a prefix-sum over block_indices");
    }

    // Per-subsequence: the provided blocks must cover past + scheduled tokens,
    // and max_context_len bounds every context.
    const auto block_size = static_cast<int64_t>(m_compiled_model->m_block_size);
    for (int64_t s = 0; s < d.n_seqs; ++s) {
        const auto ctx_after = past[s] + (sub[s + 1] - sub[s]);
        expect(past[s] >= 0, "negative past_lens entry");
        expect(mcl >= ctx_after, "max_context_len < a subsequence's context length");
        if (has_block_table && block_size > 0) {
            expect((d.block_indices_begins[s + 1] - d.block_indices_begins[s]) * block_size >= ctx_after,
                   "block_indices do not cover a subsequence's context");
        }
    }

    // Gather contract: sampled_tokens_indices picks which flat token rows get
    // logits; an empty selection is legal (intermediate prefill chunks).
    if (m_inner_inputs.count("sampled_tokens_indices") > 0) {
        d.sampled_tokens_indices = as_i64_vec(get("sampled_tokens_indices"));
        for (auto idx : d.sampled_tokens_indices) {
            expect(idx >= 0 && idx < d.n_tokens, "sampled_tokens_indices out of token range");
        }
    }

    LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << d.n_seqs << " subsequence(s), " << d.n_tokens << " token(s), "
                             << d.sampled_tokens_indices.size() << " sampled");
    return d;
}

void ov::npuw::PAInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    validate_dispatch_locked();
    m_inner_request->infer();
    ++m_dispatch_idx;
}

ov::SoPtr<ov::ITensor> ov::npuw::PAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inner_request->get_tensor(map_port_locked(port));
}

void ov::npuw::PAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                          const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_inner_request->set_tensor(map_port_locked(port), tensor);
}

void ov::npuw::PAInferRequest::check_tensors() const {
    // Tensors live in the inner request, so the base-class check over this
    // level's (empty) tensor storage must not run. The inner request performs
    // the same element-type/shape validation on its own tensors during infer().
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::PAInferRequest::query_state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inner_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::PAInferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_inner_request->get_profiling_info();
}
