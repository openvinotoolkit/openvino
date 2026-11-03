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

// The block size the CPU PA executor and the CB pipeline both use. Not read
// off the cache ports: a u8 key cache keeps its scales in the block (dim 40).
constexpr std::size_t kBlockSize = 32u;

constexpr std::array<std::size_t, 3> kChunkSizes = {1024u, 128u, 1u};

// The PA op takes its controls as i32; the lm_head gather index is i64.
template <typename T>
std::vector<T> read_vec(const ov::SoPtr<ov::ITensor>& tensor) {
    const auto* data = tensor->data<T>();
    return {data, data + tensor->get_size()};
}

template <typename T>
ov::SoPtr<ov::ITensor> make_vec_tensor(const std::vector<T>& vals) {
    auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::from<T>(), ov::Shape{vals.size()}));
    std::copy(vals.begin(), vals.end(), tensor->data<T>());
    return tensor;
}

// The inputs run_chunk can rebase per chunk. A model with any other input
// (embeddings, M-RoPE, per-layer block tables, eviction) runs every dispatch 1:1.
bool is_chunkable_pa_model(const std::shared_ptr<ov::Model>& model) {
    static const std::unordered_set<std::string> supported = {"input_ids",
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
        if (supported.count(name) == 0) {
            return false;
        }
        seen.insert(name);
    }
    for (const char* required : {"input_ids",
                                 "position_ids",
                                 "past_lens",
                                 "subsequence_begins",
                                 "block_indices",
                                 "block_indices_begins",
                                 "max_context_len",
                                 "sampled_tokens_indices"}) {
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
                                                       std::size_t chunk_size) {
    for (const char* name : {"input_ids", "position_ids"}) {
        const auto& rank = base_model->input(name).get_partial_shape().rank();
        OPENVINO_ASSERT(rank.is_static() && rank.get_length() == 1, "PA: '", name, "' is not a 1-D token stream");
    }
    auto derived = base_model->clone();
    derived->reshape({{"input_ids", ov::PartialShape{static_cast<int64_t>(chunk_size)}},
                      {"position_ids", ov::PartialShape{static_cast<int64_t>(chunk_size)}}});
    derived->set_friendly_name(base_model->get_friendly_name() + "_pa_token_" + std::to_string(chunk_size));
    return derived;
}

std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> compile_pa_semi_static_variants(
    const std::shared_ptr<ov::Model>& base_model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const std::string& device,
    const ov::AnyMap& inner_config) {
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> variants;
    for (const auto chunk_size : kChunkSizes) {
        auto derived = derive_pa_semi_static_model(base_model, chunk_size);
        auto compiled = plugin->get_core()->compile_model(derived, device, inner_config);
        OPENVINO_ASSERT(compiled != nullptr,
                        "PA semi-static derivation failed to compile chunk_size=",
                        chunk_size,
                        " on ",
                        device);
        LOG_INFO("PA: compiled semi-static variant chunk_size=" << chunk_size << " on " << device);
        variants.emplace(chunk_size, std::move(compiled));
    }

    return variants;
}

}  // anonymous namespace

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(nullptr, plugin) {  // ports are the inner model's, see inputs()
    // An env var, not an option, so it stays out of user configs. GPU would also
    // need the remote context forwarded for the pipeline's cache allocation.
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    const std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    OPENVINO_ASSERT(device == "CPU",
                    "The PagedAttention fallback device is CPU for now, got OPENVINO_NPUW_PA_DEVICE=",
                    device);

    bool has_past_lens = false, has_cache = false;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        has_past_lens |= (name == "past_lens");
        has_cache |= ov::npuw::util::is_pa_kv_cache_name(name);
    }
    OPENVINO_ASSERT(has_past_lens && has_cache,
                    "PACompiledModel expects the continuous-batching PA model "
                    "(past_lens + key_cache/value_cache inputs)");

    // NPU* keys configure this plugin and DEVICE_ID names an NPU device; the
    // executing device would reject both. Everything else is forwarded.
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

    if (is_chunkable_pa_model(model)) {
        m_semi_static_models = compile_pa_semi_static_variants(model, plugin, device, inner_config);
    } else {
        LOG_INFO("PA: model is outside the chunkable flat-token contract; every dispatch runs 1:1");
    }
    LOG_INFO("PA: " << m_semi_static_models.size() << " semi-static variant(s) on " << device);
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
    // A clear error instead of the executing device's "unsupported property".
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            OPENVINO_THROW("PACompiledModel: '", key, "' cannot be changed after the model is compiled");
        }
    }
    m_compiled_model->set_property(properties);
}

ov::Any ov::npuw::PACompiledModel::get_property(const std::string& name) const {
    // Everything but NPUW_PA is the executing device's, including
    // execution_devices, which the CB pipeline reads to pick its block size.
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    if (name == ov::supported_properties.name()) {
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
    return std::make_shared<PAInferRequest>(self, std::move(inner_request), m_semi_static_models);
}

ov::npuw::PAInferRequest::PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                         ov::SoPtr<ov::IAsyncInferRequest> inner_request,
                                         const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants)
    : ov::ISyncInferRequest(compiled_model),
      m_inner_request(std::move(inner_request)) {
    for (const auto& input : get_inputs()) {
        m_inputs_by_name.emplace(input.get_any_name(), input);
    }

    // One request per variant plus a dynamic one for residual chunks.
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
    for (const auto& [chunk_size, compiled] : variants) {
        m_chunk_requests.emplace(chunk_size, make_chunk_request(compiled));
        m_chunk_sizes.push_back(chunk_size);
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
    d.past_lens = read_vec<int32_t>(get("past_lens"));
    d.subsequence_begins = read_vec<int32_t>(get("subsequence_begins"));
    const auto mcl_vec = read_vec<int32_t>(get("max_context_len"));
    OPENVINO_ASSERT(!mcl_vec.empty(), "PA dispatch: max_context_len is not set");
    d.max_context_len = mcl_vec.front();

    // Embedding models have no input_ids; M-RoPE position_ids carry the token
    // count in the last dim.
    if (m_inputs_by_name.count("input_ids") > 0) {
        d.input_ids_size = static_cast<int64_t>(get("input_ids")->get_size());
    }
    const auto& pos_shape = get("position_ids")->get_shape();
    OPENVINO_ASSERT(!pos_shape.empty(), "PA dispatch: position_ids has no shape");
    d.position_ids_token_count = static_cast<int64_t>(pos_shape.back());

    // Eviction models carry per-layer block tables instead and run 1:1.
    if (m_inputs_by_name.count("block_indices") > 0) {
        d.has_block_table = true;
        d.block_indices = read_vec<int32_t>(get("block_indices"));
        d.block_indices_begins = read_vec<int32_t>(get("block_indices_begins"));
    }
    if (m_inputs_by_name.count("sampled_tokens_indices") > 0) {
        d.has_sampled_tokens = true;
        d.sampled_tokens_indices = read_vec<int64_t>(get("sampled_tokens_indices"));
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

    // The caller's tensors outlive this infer, so the chunk can view them.
    const auto slice = [&](const char* name, int64_t start, int64_t n) {
        return ov::npuw::util::view(inner(name), 0, static_cast<std::size_t>(start), static_cast<std::size_t>(n));
    };

    set("input_ids", slice("input_ids", global_start, n_chunk_tokens));
    set("position_ids", slice("position_ids", global_start, n_chunk_tokens));

    // One subsequence, past its first seq_offset tokens, with its full block table.
    set("past_lens", make_vec_tensor<int32_t>({static_cast<int32_t>(d.past_lens[seq] + seq_offset)}));
    set("subsequence_begins", make_vec_tensor<int32_t>({0, static_cast<int32_t>(n_chunk_tokens)}));
    const auto blocks_begin = d.block_indices_begins[seq];
    const auto n_seq_blocks = d.block_indices_begins[seq + 1] - blocks_begin;
    set("block_indices", slice("block_indices", blocks_begin, n_seq_blocks));
    set("block_indices_begins", make_vec_tensor<int32_t>({0, n_seq_blocks}));
    if (m_inputs_by_name.count("score_aggregation_window") > 0) {
        set("score_aggregation_window", slice("score_aggregation_window", seq, 1));
    }

    set("max_context_len", inner("max_context_len"));
    for (const auto& [name, port] : chunk.inputs) {
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
            chunk.request->set_tensor(port, m_inner_request->get_tensor(m_inputs_by_name.at(name)));
        }
    }

    // Sampled rows in this chunk, with their row in the caller's output.
    std::vector<int64_t> local_sti;
    std::vector<std::size_t> out_rows;
    for (std::size_t i = 0; i < d.sampled_tokens_indices.size(); ++i) {
        const auto g = d.sampled_tokens_indices[i];
        if (g >= global_start && g < global_start + n_chunk_tokens) {
            local_sti.push_back(g - global_start);
            out_rows.push_back(i);
        }
    }
    set("sampled_tokens_indices", make_vec_tensor<int64_t>(local_sti));

    // The logits port is dynamic (one row per sampled token) and NPUW sizes
    // unset outputs from the port, so set an exact-sized tensor.
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
            // Largest variant that fits; the 1-token one only for a single
            // remaining token. Anything else goes to the dynamic model.
            std::size_t pick = 0u;
            for (const auto& [chunk_size, _] : m_chunk_requests) {
                if (static_cast<int64_t>(chunk_size) <= remaining && (chunk_size > 1u || remaining == 1)) {
                    pick = chunk_size;
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
        const auto tensor = outputs && m_serve_chunked_logits ? m_chunked_logits : m_inner_request->get_tensor(port);
        LOG_VERB(name << ": " << ov::npuw::util::TensorBrief{tensor});
    }
}

void ov::npuw::PAInferRequest::infer() {
    log_dispatch_io(/*outputs=*/false);
    const auto dispatch = parse_dispatch();
    pa::validate_dispatch(dispatch, kBlockSize, m_dispatch_idx);
    LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << dispatch.sequences() << " subsequence(s), "
                             << dispatch.tokens() << " token(s), " << dispatch.sampled_tokens_indices.size()
                             << " sampled");
    if (pa::variants_serve(dispatch, m_chunk_sizes)) {
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
    // Tensors live in the inner request, which checks them itself.
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::PAInferRequest::query_state() const {
    return m_inner_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::PAInferRequest::get_profiling_info() const {
    return m_inner_request->get_profiling_info();
}
