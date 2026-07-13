// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
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

std::string abbrev(const std::vector<int64_t>& vals, std::size_t limit = 8) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < std::min(vals.size(), limit); ++i) {
        out << (i ? ", " : "") << vals[i];
    }
    if (vals.size() > limit) {
        out << ", ... " << vals.size() - limit << " more";
    }
    out << ']';
    return out.str();
}

bool try_fix_token_dim(const std::string& name, ov::PartialShape& shape, std::size_t token_dim) {
    if (!shape.rank().is_static()) {
        return false;
    }

    const auto rank = shape.rank().get_length();
    const auto set_dim = [&](std::size_t idx) {
        if (idx >= shape.size()) {
            return false;
        }
        shape[idx] = static_cast<int64_t>(token_dim);
        return true;
    };

    if (name == "input_ids" || name == "token_type_ids" || name == "position_ids") {
        return set_dim(static_cast<std::size_t>(rank - 1));
    }
    if (name == "inputs_embeds" || name == "per_layer_inputs") {
        if (rank < 2) {
            return false;
        }
        return set_dim(static_cast<std::size_t>(rank - 2));
    }
    return false;
}

// A fresh vector tensor with the chunk port's (integer) element type.
ov::SoPtr<ov::ITensor> make_ctrl_tensor(const ov::Output<const ov::Node>& port, const std::vector<int64_t>& vals) {
    const auto type = port.get_element_type();
    auto tensor = ov::get_tensor_impl(ov::Tensor(type, ov::Shape{vals.size()}));
    if (type == ov::element::i32) {
        std::transform(vals.begin(), vals.end(), tensor->data<int32_t>(), [](int64_t v) {
            return static_cast<int32_t>(v);
        });
    } else if (type == ov::element::i64) {
        std::copy(vals.begin(), vals.end(), tensor->data<int64_t>());
    } else {
        OPENVINO_THROW("PA: unexpected element type ", type, " for a control tensor");
    }
    return tensor;
}

// [start, start + n) of a 1-D tensor, copied into a fresh same-typed tensor.
ov::SoPtr<ov::ITensor> slice_1d(const ov::SoPtr<ov::ITensor>& src, int64_t start, int64_t n) {
    auto out = ov::get_tensor_impl(ov::Tensor(src->get_element_type(), ov::Shape{static_cast<std::size_t>(n)}));
    const auto esize = src->get_element_type().size();
    const auto* base = static_cast<const uint8_t*>(src->data());
    std::memcpy(out->data(), base + static_cast<std::size_t>(start) * esize, static_cast<std::size_t>(n) * esize);
    return out;
}

std::shared_ptr<ov::Model> derive_pa_semi_static_model(const std::shared_ptr<ov::Model>& base_model,
                                                       std::size_t token_dim) {
    auto derived = base_model->clone();
    std::map<std::string, ov::PartialShape> new_shapes;
    bool touched = false;

    for (const auto& input : derived->inputs()) {
        auto shape = input.get_partial_shape();
        if (try_fix_token_dim(input.get_any_name(), shape, token_dim)) {
            new_shapes.emplace(input.get_any_name(), std::move(shape));
            touched = true;
        }
    }

    OPENVINO_ASSERT(touched,
                    "PA semi-static derivation (token_dim=",
                    token_dim,
                    ") did not find token-driven inputs to reshape");
    derived->reshape(new_shapes);
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
    const auto device_key = std::string(::intel_npu::NPUW_PA_DEVICE::key());
    auto device_it = properties.find(device_key);
    std::string device = device_it != properties.end() ? device_it->second.as<std::string>()
                                                       : std::string(::intel_npu::NPUW_PA_DEVICE::defaultValue());

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

    auto semi_static_compiled = compile_pa_semi_static_variants(model, plugin, device, inner_config);
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
    LOG_BLOCK();
    // Trace the compiled signature -- the model expectations this story is
    // about. The device fixes the KV cache geometry at compile time.
    std::size_t num_cache_inputs = 0u;
    for (const auto& input : m_compiled_model->inputs()) {
        const auto& name = input.get_any_name();
        if (is_kv_cache_name(name)) {
            ++num_cache_inputs;
            if (num_cache_inputs > 2) {
                continue;
            }
        }
        LOG_INFO("in  " << name << " " << input.get_element_type() << " " << input.get_partial_shape());
        if (name == "key_cache.0") {
            const auto& shape = input.get_partial_shape();
            // [num_blocks (dyn), kv_heads, block_size, head_size]
            if (shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static()) {
                m_block_size = static_cast<std::size_t>(shape[2].get_length());
            }
        }
    }
    if (num_cache_inputs > 2) {
        LOG_INFO("in  ... " << num_cache_inputs << " key_cache/value_cache inputs total (" << num_cache_inputs / 2
                            << " layers)");
    }
    for (const auto& output : m_compiled_model->outputs()) {
        LOG_INFO("out " << output.get_any_name() << " " << output.get_element_type() << " "
                        << output.get_partial_shape());
    }
    LOG_INFO("KV block_size fixed by " << m_device << ": " << m_block_size);
    if (!m_semi_static_models.empty()) {
        std::ostringstream variants;
        for (const auto& [token_dim, _] : m_semi_static_models) {
            variants << (variants.tellp() > 0 ? ", " : "") << token_dim;
        }
        LOG_INFO("PA: semi-static variants ready (token_dim): " << variants.str());
    }
}

void ov::npuw::PACompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not support export_model() -- PA Stage 0");
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
    // The PA-level knobs are answered here; everything else is the executing
    // device's business (notably ov::execution_devices, which the CB pipeline
    // queries to pick its block size).
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    if (name == std::string(::intel_npu::NPUW_PA_DEVICE::key())) {
        return m_device;
    }
    if (name == ov::supported_properties.name()) {
        // Keep the property surface self-consistent: the inner device's list
        // plus the PA keys answered above.
        auto props = m_compiled_model->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(std::string(::intel_npu::NPUW_PA::key()), ov::PropertyMutability::RO);
        props.emplace_back(std::string(::intel_npu::NPUW_PA_DEVICE::key()), ov::PropertyMutability::RO);
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

    // Chunked execution: one request per semi-static variant plus a dynamic
    // request for residual chunks. They run against the same paged KV cache
    // tensors as the inner request, so they can be prepared upfront.
    const auto make_chunk_request = [](const ov::SoPtr<ov::ICompiledModel>& compiled) {
        ChunkRequest chunk;
        chunk.request = compiled->create_infer_request();
        OPENVINO_ASSERT(chunk.request != nullptr, "PA chunk model requires a valid infer request");
        for (const auto& input : compiled->inputs()) {
            chunk.inputs.emplace(input.get_any_name(), input);
        }
        for (const auto& output : compiled->outputs()) {
            if (output.get_any_name() == "logits") {
                chunk.logits = output;
            }
        }
        return chunk;
    };
    for (const auto& [token_dim, compiled] : m_compiled_model->m_semi_static_models) {
        m_chunk_requests.emplace(token_dim, make_chunk_request(compiled));
    }
    m_tail_request = make_chunk_request(m_compiled_model->m_compiled_model);
    for (const auto& output : m_compiled_model->outputs()) {
        if (output.get_any_name() == "logits") {
            m_logits_node = output.get_node();
        }
    }
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

    Dispatch d;
    d.past_lens = as_i64_vec(get("past_lens"));
    d.subsequence_begins = as_i64_vec(get("subsequence_begins"));
    d.block_indices = as_i64_vec(get("block_indices"));
    d.block_indices_begins = as_i64_vec(get("block_indices_begins"));
    const auto& past = d.past_lens;
    const auto& sub = d.subsequence_begins;
    const auto& bib = d.block_indices_begins;
    const auto n_blocks = static_cast<int64_t>(d.block_indices.size());
    const auto mcl_vec = as_i64_vec(get("max_context_len"));
    const auto mcl = mcl_vec.empty() ? int64_t{-1} : mcl_vec.front();
    const auto n_seqs = static_cast<int64_t>(past.size());
    // subsequence_begins is the source of truth for the flat token dimension.
    // input_ids is absent on embedding-input models (inputs_embeds), so it is
    // only cross-checked when present; position_ids may be multi-dimensional
    // (M-RoPE), so its token count is the last shape dim.
    const auto n_tokens = sub.empty() ? int64_t{0} : sub.back();
    d.max_context_len = mcl;
    d.n_seqs = n_seqs;
    d.n_tokens = n_tokens;

    std::vector<std::string> violations;
    const auto expect = [&](bool cond, const std::string& what) {
        if (!cond) {
            violations.push_back(what);
        }
    };

    expect(!mcl_vec.empty(), "max_context_len is not set");
    if (m_inner_inputs.count("input_ids") > 0) {
        expect(static_cast<int64_t>(get("input_ids")->get_size()) == n_tokens,
               "input_ids size != subsequence_begins token count");
    }
    const auto& pos_shape = get("position_ids")->get_shape();
    expect(!pos_shape.empty() && static_cast<int64_t>(pos_shape.back()) == n_tokens,
           "position_ids last dim != subsequence_begins token count");
    expect(static_cast<int64_t>(sub.size()) == n_seqs + 1, "subsequence_begins size != past_lens size + 1");
    expect(!sub.empty() && sub.front() == 0, "subsequence_begins does not start at 0");
    expect(std::is_sorted(sub.begin(), sub.end()) && std::adjacent_find(sub.begin(), sub.end()) == sub.end(),
           "subsequence_begins is not strictly increasing");
    expect(static_cast<int64_t>(bib.size()) == n_seqs + 1, "block_indices_begins size != past_lens size + 1");
    expect(!bib.empty() && bib.front() == 0 && bib.back() == n_blocks && std::is_sorted(bib.begin(), bib.end()),
           "block_indices_begins is not a prefix-sum over block_indices");

    // Per-subsequence: the provided blocks must cover past + scheduled tokens,
    // and max_context_len bounds every context.
    const bool verbose = ov::npuw::get_log_level() >= ov::npuw::LogLevel::Verbose;
    std::ostringstream kinds;
    const auto block_size = static_cast<int64_t>(m_compiled_model->m_block_size);
    for (int64_t s = 0; s + 1 < static_cast<int64_t>(sub.size()) && s < n_seqs; ++s) {
        const auto scheduled = sub[s + 1] - sub[s];
        const auto ctx_after = past[s] + scheduled;
        expect(past[s] >= 0, "negative past_lens entry");
        expect(mcl >= ctx_after, "max_context_len < a subsequence's context length");
        if (block_size > 0 && s + 1 < static_cast<int64_t>(bib.size())) {
            expect((bib[s + 1] - bib[s]) * block_size >= ctx_after,
                   "block_indices do not cover a subsequence's context");
        }
        if (verbose) {
            kinds << (s ? ", " : "") << (past[s] == 0 ? "prefill" : (scheduled == 1 ? "decode" : "chunked-continue"));
        }
    }

    // Gather contract: sampled_tokens_indices picks which flat token rows get
    // logits; an empty selection is legal (intermediate prefill chunks).
    if (m_inner_inputs.count("sampled_tokens_indices") > 0) {
        d.has_sti = true;
        d.sampled_tokens_indices = as_i64_vec(get("sampled_tokens_indices"));
        for (auto idx : d.sampled_tokens_indices) {
            expect(idx >= 0 && idx < n_tokens, "sampled_tokens_indices out of token range");
        }
    }

    if (verbose) {
        LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << n_seqs << " subsequence(s) [" << kinds.str() << "], "
                                 << n_tokens << " token(s)");
        LOG_VERB("  past_lens              " << abbrev(past));
        LOG_VERB("  subsequence_begins     " << abbrev(sub));
        LOG_VERB("  block_indices          " << n_blocks << " entries, begins " << abbrev(bib));
        LOG_VERB("  max_context_len        " << mcl);
        if (d.has_sti) {
            LOG_VERB("  sampled_tokens_indices " << abbrev(d.sampled_tokens_indices) << " -> "
                                                 << d.sampled_tokens_indices.size() << " logits row(s)");
        }
    }

    if (!violations.empty()) {
        std::ostringstream all;
        for (const auto& v : violations) {
            all << "\n  " << v;
        }
        OPENVINO_THROW("PA dispatch #", m_dispatch_idx, " violates the PA model expectations:", all.str());
    }
    return d;
}

void ov::npuw::PAInferRequest::validate_output_locked(int64_t expected_logits_rows) {
    if (expected_logits_rows < 0) {
        return;
    }
    for (const auto& output : m_inner_request->get_compiled_model()->outputs()) {
        if (output.get_any_name() != "logits") {
            continue;
        }
        const auto rows = static_cast<int64_t>(m_inner_request->get_tensor(output)->get_shape().at(0));
        OPENVINO_ASSERT(rows == expected_logits_rows,
                        "PA dispatch #",
                        m_dispatch_idx,
                        ": logits rows (",
                        rows,
                        ") != sampled_tokens_indices count (",
                        expected_logits_rows,
                        ")");
    }
}

bool ov::npuw::PAInferRequest::can_chunk_locked(const Dispatch& d) const {
    // The generation case needs the 1-token model; without it (or without
    // anything to gather), the base dynamic model handles the dispatch 1:1.
    if (m_chunk_requests.empty() || m_chunk_requests.count(1u) == 0 || m_logits_node == nullptr) {
        return false;
    }
    if (!d.has_sti || d.n_tokens <= 0) {
        return false;
    }
    for (const auto& [_, chunk] : m_chunk_requests) {
        if (chunk.logits.get_node() == nullptr) {
            return false;
        }
    }
    // Only the plain flat-token LLM contract is chunked; embedding inputs,
    // M-RoPE position_ids and other extras fall back to the 1:1 path.
    static const std::unordered_set<std::string> known = {"input_ids",
                                                          "position_ids",
                                                          "past_lens",
                                                          "subsequence_begins",
                                                          "block_indices",
                                                          "block_indices_begins",
                                                          "max_context_len",
                                                          "score_aggregation_window",
                                                          "sampled_tokens_indices"};
    for (const auto& [name, port] : m_inner_inputs) {
        if (!is_kv_cache_name(name) && known.count(name) == 0) {
            return false;
        }
    }
    if (m_inner_inputs.count("input_ids") == 0 || m_inner_inputs.count("position_ids") == 0) {
        return false;
    }
    if (m_inner_request->get_tensor(m_inner_inputs.at("input_ids"))->get_shape().size() != 1 ||
        m_inner_request->get_tensor(m_inner_inputs.at("position_ids"))->get_shape().size() != 1) {
        return false;
    }
    // A single logits output whose per-row geometry is static, so the result
    // tensor can be allocated upfront and filled row by row.
    const auto& outputs = m_inner_request->get_compiled_model()->outputs();
    if (outputs.size() != 1 || outputs.front().get_any_name() != "logits") {
        return false;
    }
    const auto& lshape = outputs.front().get_partial_shape();
    if (lshape.rank().is_dynamic() || lshape.rank().get_length() != 3 || lshape[1].is_dynamic() ||
        lshape[2].is_dynamic()) {
        return false;
    }
    return true;
}

void ov::npuw::PAInferRequest::run_chunk_locked(ChunkRequest& chunk,
                                                const Dispatch& d,
                                                int64_t seq,
                                                int64_t seq_offset,
                                                int64_t n_chunk_tokens) {
    const auto global_start = d.subsequence_begins[seq] + seq_offset;
    const auto set = [&](const char* name, const ov::SoPtr<ov::ITensor>& tensor) {
        auto it = chunk.inputs.find(name);
        OPENVINO_ASSERT(it != chunk.inputs.end(), "PA chunk model has no '", name, "' input");
        chunk.request->set_tensor(it->second, tensor);
    };
    const auto inner = [&](const char* name) {
        return m_inner_request->get_tensor(m_inner_inputs.at(name));
    };

    // Token-driven inputs: this chunk's slice of the caller's flat stream.
    set("input_ids", slice_1d(inner("input_ids"), global_start, n_chunk_tokens));
    set("position_ids", slice_1d(inner("position_ids"), global_start, n_chunk_tokens));

    // Per-subsequence controls, rebased to a single subsequence that has
    // already seen seq_offset of its scheduled tokens. The block table is the
    // subsequence's full table: context stays dynamic, positions address it.
    set("past_lens", make_ctrl_tensor(chunk.inputs.at("past_lens"), {d.past_lens[seq] + seq_offset}));
    set("subsequence_begins", make_ctrl_tensor(chunk.inputs.at("subsequence_begins"), {0, n_chunk_tokens}));
    const auto blocks_begin = d.block_indices_begins[seq];
    const auto n_seq_blocks = d.block_indices_begins[seq + 1] - blocks_begin;
    set("block_indices", slice_1d(inner("block_indices"), blocks_begin, n_seq_blocks));
    set("block_indices_begins", make_ctrl_tensor(chunk.inputs.at("block_indices_begins"), {0, n_seq_blocks}));
    if (m_inner_inputs.count("score_aggregation_window") > 0) {
        set("score_aggregation_window", slice_1d(inner("score_aggregation_window"), seq, 1));
    }

    // The whole-batch max_context_len still bounds this chunk's context, and
    // the paged KV cache pools are shared as-is.
    set("max_context_len", inner("max_context_len"));
    for (const auto& [name, port] : chunk.inputs) {
        if (is_kv_cache_name(name)) {
            chunk.request->set_tensor(port, m_inner_request->get_tensor(m_inner_inputs.at(name)));
        }
    }

    // Sampled rows falling into this chunk, remembered with their position in
    // the caller's sampled_tokens_indices order.
    std::vector<int64_t> local_sti;
    std::vector<std::size_t> out_rows;
    for (std::size_t i = 0; i < d.sampled_tokens_indices.size(); ++i) {
        const auto g = d.sampled_tokens_indices[i];
        if (g >= global_start && g < global_start + n_chunk_tokens) {
            local_sti.push_back(g - global_start);
            out_rows.push_back(i);
        }
    }
    set("sampled_tokens_indices", make_ctrl_tensor(chunk.inputs.at("sampled_tokens_indices"), local_sti));

    chunk.request->infer();

    if (out_rows.empty()) {
        return;
    }
    const auto out = chunk.request->get_tensor(chunk.logits);
    OPENVINO_ASSERT(out->get_shape().at(0) == out_rows.size(),
                    "PA chunk produced ",
                    out->get_shape().at(0),
                    " logits row(s), expected ",
                    out_rows.size());
    const auto& oshape = m_chunked_logits->get_shape();
    const auto row_bytes = oshape.at(1) * oshape.at(2) * m_chunked_logits->get_element_type().size();
    const auto* src = static_cast<const uint8_t*>(out->data());
    auto* dst = static_cast<uint8_t*>(m_chunked_logits->data());
    for (std::size_t j = 0; j < out_rows.size(); ++j) {
        std::memcpy(dst + out_rows[j] * row_bytes, src + j * row_bytes, row_bytes);
    }
}

void ov::npuw::PAInferRequest::infer_chunked_locked(const Dispatch& d) {
    // One logits row per sampled token, in the caller's order.
    const auto& logits_port = m_inner_request->get_compiled_model()->outputs().front();
    const auto& lshape = logits_port.get_partial_shape();
    m_chunked_logits = ov::get_tensor_impl(ov::Tensor(logits_port.get_element_type(),
                                                      ov::Shape{d.sampled_tokens_indices.size(),
                                                                static_cast<std::size_t>(lshape[1].get_length()),
                                                                static_cast<std::size_t>(lshape[2].get_length())}));

    const bool verbose = ov::npuw::get_log_level() >= ov::npuw::LogLevel::Verbose;
    std::ostringstream plan;

    for (int64_t s = 0; s < d.n_seqs; ++s) {
        const auto seq_len = d.subsequence_begins[s + 1] - d.subsequence_begins[s];
        int64_t off = 0;
        if (verbose) {
            plan << (s ? "; " : "") << "seq" << s << "=";
        }
        while (off < seq_len) {
            const auto remaining = seq_len - off;
            // Largest variant that fits; the 1-token model is only right when
            // exactly one token remains (the generation case). Everything
            // else that no variant fits goes through the dynamic model.
            std::size_t pick = 0u;
            for (const auto& [token_dim, _] : m_chunk_requests) {
                if (static_cast<int64_t>(token_dim) <= remaining && (token_dim > 1u || remaining == 1)) {
                    pick = token_dim;
                    break;
                }
            }
            auto& chunk = pick ? m_chunk_requests.at(pick) : m_tail_request;
            const auto n = pick ? static_cast<int64_t>(pick) : remaining;
            if (verbose) {
                plan << (off ? "+" : "") << (pick ? "" : "dyn:") << n;
            }
            run_chunk_locked(chunk, d, s, off, n);
            off += n;
        }
    }
    if (verbose) {
        LOG_VERB("PA dispatch #" << m_dispatch_idx << ": chunked " << plan.str());
    }
}

void ov::npuw::PAInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto dispatch = validate_dispatch_locked();
    if (can_chunk_locked(dispatch)) {
        infer_chunked_locked(dispatch);
        m_serve_chunked_logits = true;
    } else {
        m_serve_chunked_logits = false;
        m_inner_request->infer();
        validate_output_locked(dispatch.has_sti ? static_cast<int64_t>(dispatch.sampled_tokens_indices.size())
                                                : int64_t{-1});
    }
    ++m_dispatch_idx;
}

ov::SoPtr<ov::ITensor> ov::npuw::PAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_serve_chunked_logits && port.get_node() == m_logits_node) {
        return m_chunked_logits;
    }
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
