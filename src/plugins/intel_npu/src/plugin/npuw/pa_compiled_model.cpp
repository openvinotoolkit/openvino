// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

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
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
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

// Compact one-line digest of a tensor for the per-dispatch I/O trace:
// element type, shape, then the values for small tensors or min/max/mean for
// large ones. KV cache pools pass with_data=false -- the data is the whole
// paged cache, so only the geometry is shown.
std::string tensor_brief(const ov::SoPtr<ov::ITensor>& tensor, bool with_data = true) {
    std::ostringstream os;
    os << tensor->get_element_type() << " " << tensor->get_shape();
    const auto type = tensor->get_element_type();
    const auto n = tensor->get_size();
    const bool readable =
        type == ov::element::f32 || type == ov::element::f16 || type == ov::element::i32 || type == ov::element::i64;
    if (!with_data || !readable || n == 0) {
        return os.str();
    }
    const auto value_at = [&](std::size_t i) -> double {
        if (type == ov::element::f32) {
            return tensor->data<float>()[i];
        }
        if (type == ov::element::f16) {
            return static_cast<float>(tensor->data<ov::float16>()[i]);
        }
        if (type == ov::element::i32) {
            return tensor->data<int32_t>()[i];
        }
        return static_cast<double>(tensor->data<int64_t>()[i]);
    };
    constexpr std::size_t kMaxInline = 16u;
    if (n <= kMaxInline) {
        os << " {";
        for (std::size_t i = 0; i < n; ++i) {
            os << (i ? ", " : "") << value_at(i);
        }
        os << "}";
    } else {
        auto lo = value_at(0), hi = lo, sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto v = value_at(i);
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            sum += v;
        }
        os << " min=" << lo << " max=" << hi << " mean=" << sum / static_cast<double>(n);
    }
    return os.str();
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

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(nullptr, plugin) {  // I/O comes from the inner via inputs()/outputs()
    // The fallback device is an internal development knob, not a config
    // option - an env var keeps it out of user configs (and blob cache keys).
    // Only CPU is supported for now: a GPU device would also need the remote
    // context forwarded for the pipeline's cache allocation.
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    const std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    OPENVINO_ASSERT(device == "CPU",
                    "The PagedAttention fallback device is CPU for now, got OPENVINO_NPUW_PA_DEVICE=",
                    device);

    // Sanity: this must be the model the CB pipeline deploys -- PA control
    // inputs plus a paged KV cache.
    bool has_past_lens = false, has_cache = false;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        has_past_lens |= (name == "past_lens");
        has_cache |= ov::npuw::util::is_pa_kv_cache_name(name);
    }
    OPENVINO_ASSERT(has_past_lens && has_cache,
                    "PACompiledModel expects the continuous-batching PA model "
                    "(past_lens + key_cache/value_cache inputs)");

    // The 1:1 part: the model is compiled exactly as received. NPUW_*,
    // NPU_USE_NPUW and NPU_* keys are this plugin's configuration and must not
    // reach the executing device (which would reject them as unsupported);
    // everything else (e.g. KV_CACHE_PRECISION, performance hints) is the
    // executing device's business and is forwarded. DEVICE_ID names an NPU
    // device (e.g. NPU.3600), so it stays behind as well: the fallback device
    // would reject an id it doesn't have.
    ov::AnyMap inner_config;
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU") || key == ov::device::id.name()) {
            continue;
        }
        inner_config.emplace(key, value);
    }

    LOG_INFO("PA: compiling the dynamic PA model 1:1 on " << device);
    m_compiled_model = plugin->get_core()->compile_model(model, device, inner_config);
    OPENVINO_ASSERT(m_compiled_model != nullptr, "PACompiledModel requires a valid inner compiled model");

    // The device fixes the KV cache geometry at compile time; remember the
    // block size for validating block-table coverage per dispatch. Identical
    // across layers by construction, so the first cache input answers it.
    for (const auto& input : m_compiled_model->inputs()) {
        if (ov::npuw::util::is_pa_key_cache_name(input.get_any_name())) {
            const auto& shape = input.get_partial_shape();
            // [num_blocks (dyn), kv_heads, block_size, head_size]
            if (shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static()) {
                m_block_size = static_cast<std::size_t>(shape[2].get_length());
            }
            break;
        }
    }

    // The semi-static variants only make sense for the plain flat-token LLM
    // contract; anything else (VLM, M-RoPE, per-layer block tables, ...) runs
    // 1:1 on the dynamic model, so don't spend compile time on variants.
    if (is_chunkable_pa_model(model)) {
        m_semi_static_models = compile_pa_semi_static_variants(model, plugin, device, inner_config);
    } else {
        LOG_INFO("PA: model is outside the chunkable flat-token contract; every dispatch runs 1:1");
    }
    LOG_INFO("PA: KV block_size fixed by " << device << ": " << m_block_size << "; " << m_semi_static_models.size()
                                           << " semi-static variant(s)");
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::PACompiledModel::inputs() const {
    return m_compiled_model->inputs();
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::PACompiledModel::outputs() const {
    return m_compiled_model->outputs();
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
    auto self = std::static_pointer_cast<const ov::ICompiledModel>(shared_from_this());
    auto inner_request = m_compiled_model->create_infer_request();
    OPENVINO_ASSERT(inner_request != nullptr, "PACompiledModel requires a valid inner infer request");
    return std::make_shared<PAInferRequest>(self, std::move(inner_request), m_block_size, m_semi_static_models);
}

ov::npuw::PAInferRequest::PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                         ov::SoPtr<ov::IAsyncInferRequest> inner_request,
                                         std::size_t block_size,
                                         const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants)
    : ov::ISyncInferRequest(compiled_model),
      m_inner_request(std::move(inner_request)),
      m_block_size(block_size) {
    for (const auto& input : get_inputs()) {
        m_inputs_by_name.emplace(input.get_any_name(), input);
    }

    // Chunked execution: one request per semi-static variant plus a dynamic
    // request for residual chunks. They run against the same paged KV cache
    // tensors as the inner request, so they can be prepared upfront. The
    // variants only exist for chunkable models (single logits output).
    const auto make_chunk_request = [](const auto& compiled) {
        ChunkRequest chunk;
        chunk.request = compiled->create_infer_request();
        OPENVINO_ASSERT(chunk.request != nullptr, "PA chunk model requires a valid infer request");
        for (const auto& input : compiled->inputs()) {
            chunk.inputs.emplace(input.get_any_name(), input);
        }
        chunk.logits = compiled->outputs().front();
        return chunk;
    };
    for (const auto& [token_dim, compiled] : variants) {
        m_chunk_requests.emplace(token_dim, make_chunk_request(compiled));
        m_variant_token_dims.push_back(token_dim);
    }
    if (!m_chunk_requests.empty()) {
        m_tail_request = make_chunk_request(m_inner_request->get_compiled_model());
        m_logits_node = get_outputs().front().get_node();
    }
}

ov::npuw::pa::Dispatch ov::npuw::PAInferRequest::parse_dispatch() const {
    const auto get = [&](const char* name) {
        auto it = m_inputs_by_name.find(name);
        OPENVINO_ASSERT(it != m_inputs_by_name.end(), "PA model has no '", name, "' input");
        return m_inner_request->get_tensor(it->second);
    };

    pa::Dispatch d;
    d.past_lens = as_i64_vec(get("past_lens"));
    d.subsequence_begins = as_i64_vec(get("subsequence_begins"));
    const auto mcl_vec = as_i64_vec(get("max_context_len"));
    OPENVINO_ASSERT(!mcl_vec.empty(), "PA dispatch: max_context_len is not set");
    d.max_context_len = mcl_vec.front();

    // input_ids is absent on embedding-input models (inputs_embeds);
    // position_ids may be multi-dimensional (M-RoPE), so its token count is
    // the last shape dim.
    if (m_inputs_by_name.count("input_ids") > 0) {
        d.input_ids_size = static_cast<int64_t>(get("input_ids")->get_size());
    }
    const auto& pos_shape = get("position_ids")->get_shape();
    OPENVINO_ASSERT(!pos_shape.empty(), "PA dispatch: position_ids has no shape");
    d.position_ids_token_count = static_cast<int64_t>(pos_shape.back());

    // The shared block table. Cache-eviction models carry per-layer
    // block_indices.<L> inputs instead; those dispatches run 1:1 and only the
    // common controls are validated.
    if (m_inputs_by_name.count("block_indices") > 0) {
        d.has_block_table = true;
        d.block_indices = as_i64_vec(get("block_indices"));
        d.block_indices_begins = as_i64_vec(get("block_indices_begins"));
    }
    if (m_inputs_by_name.count("sampled_tokens_indices") > 0) {
        d.has_sampled_tokens = true;
        d.sampled_tokens_indices = as_i64_vec(get("sampled_tokens_indices"));
    }
    return d;
}

void ov::npuw::PAInferRequest::run_chunk(ChunkRequest& chunk,
                                         const pa::Dispatch& d,
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
        return m_inner_request->get_tensor(m_inputs_by_name.at(name));
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
    if (m_inputs_by_name.count("score_aggregation_window") > 0) {
        set("score_aggregation_window", slice_1d(inner("score_aggregation_window"), seq, 1));
    }

    // The whole-batch max_context_len still bounds this chunk's context, and
    // the paged KV cache pools are shared as-is.
    set("max_context_len", inner("max_context_len"));
    for (const auto& [name, port] : chunk.inputs) {
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
            chunk.request->set_tensor(port, m_inner_request->get_tensor(m_inputs_by_name.at(name)));
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

    // The logits row count is the number of sampled tokens, so the output port
    // stays dynamic and the executing request cannot allocate it on its own
    // (NPUW in particular sizes unset outputs from the port's static shape).
    // The row count is known right here, so pre-set an exact-sized tensor.
    const auto& oshape = m_chunked_logits->get_shape();
    const auto out = ov::get_tensor_impl(
        ov::Tensor(m_chunked_logits->get_element_type(), ov::Shape{local_sti.size(), oshape.at(1), oshape.at(2)}));
    chunk.request->set_tensor(chunk.logits, out);

    chunk.request->infer();

    if (out_rows.empty()) {
        return;
    }
    const auto row_bytes = oshape.at(1) * oshape.at(2) * m_chunked_logits->get_element_type().size();
    const auto* src = static_cast<const uint8_t*>(out->data());
    auto* dst = static_cast<uint8_t*>(m_chunked_logits->data());
    for (std::size_t j = 0; j < out_rows.size(); ++j) {
        std::memcpy(dst + out_rows[j] * row_bytes, src + j * row_bytes, row_bytes);
    }
}

void ov::npuw::PAInferRequest::infer_chunked(const pa::Dispatch& d) {
    // One logits row per sampled token, in the caller's order.
    const auto& logits_port = get_outputs().front();
    const auto& lshape = logits_port.get_partial_shape();
    m_chunked_logits = ov::get_tensor_impl(ov::Tensor(logits_port.get_element_type(),
                                                      ov::Shape{d.sampled_tokens_indices.size(),
                                                                static_cast<std::size_t>(lshape[1].get_length()),
                                                                static_cast<std::size_t>(lshape[2].get_length())}));

    const bool verbose = ov::npuw::get_log_level() >= ov::npuw::LogLevel::Verbose;
    std::ostringstream plan;

    for (int64_t s = 0; s < d.sequences(); ++s) {
        const auto seq_len = d.subsequence_begins[s + 1] - d.subsequence_begins[s];
        int64_t off = 0;
        if (verbose) {
            plan << (s ? "; " : "") << "seq" << s << "=";
        }
        while (off < seq_len) {
            const auto remaining = seq_len - off;
            // Largest variant that fits (m_chunk_requests is ordered largest
            // first); the 1-token model is only right when exactly one token
            // remains (the generation case). Everything else that no variant
            // fits goes through the dynamic model.
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
            run_chunk(chunk, d, s, off, n);
            off += n;
        }
    }
    if (verbose) {
        LOG_VERB("PA dispatch #" << m_dispatch_idx << ": chunked " << plan.str());
    }
}

void ov::npuw::PAInferRequest::log_dispatch_io(bool outputs) const {
    if (ov::npuw::get_log_level() < ov::npuw::LogLevel::Verbose) {
        return;
    }
    LOG_VERB("PA dispatch #" << m_dispatch_idx << (outputs ? " outputs:" : " inputs:"));
    LOG_BLOCK();
    for (const auto& port : outputs ? get_outputs() : get_inputs()) {
        const auto& name = port.get_any_name();
        // On the chunked path the inner request was not inferred; the result
        // lives in m_chunked_logits (the model's single output).
        const auto tensor = outputs && m_serve_chunked_logits ? m_chunked_logits : m_inner_request->get_tensor(port);
        LOG_VERB(name << ": " << tensor_brief(tensor, !ov::npuw::util::is_pa_kv_cache_name(name)));
    }
}

void ov::npuw::PAInferRequest::infer() {
    log_dispatch_io(/*outputs=*/false);
    const auto dispatch = parse_dispatch();
    pa::validate_dispatch(dispatch, m_block_size, m_dispatch_idx);
    LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << dispatch.sequences() << " subsequence(s), "
                             << dispatch.tokens() << " token(s), " << dispatch.sampled_tokens_indices.size()
                             << " sampled");
    if (pa::variants_serve(dispatch, m_variant_token_dims)) {
        infer_chunked(dispatch);
        m_serve_chunked_logits = true;
    } else {
        m_serve_chunked_logits = false;
        m_inner_request->infer();
    }
    log_dispatch_io(/*outputs=*/true);
    ++m_dispatch_idx;
}

ov::SoPtr<ov::ITensor> ov::npuw::PAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    if (m_serve_chunked_logits && port.get_node() == m_logits_node) {
        return m_chunked_logits;
    }
    return m_inner_request->get_tensor(port);
}

void ov::npuw::PAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                          const ov::SoPtr<ov::ITensor>& tensor) {
    m_inner_request->set_tensor(port, tensor);
}

void ov::npuw::PAInferRequest::check_tensors() const {
    // Tensors live in the inner request, so the base-class check over this
    // level's (empty) tensor storage must not run. The inner request performs
    // the same element-type/shape validation on its own tensors during infer().
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::PAInferRequest::query_state() const {
    return m_inner_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::PAInferRequest::get_profiling_info() const {
    return m_inner_request->get_profiling_info();
}
