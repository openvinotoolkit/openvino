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
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

// The block size the CPU PA executor and the CB pipeline both use. Not read
// off the cache ports: a u8 key cache keeps its scales in the block (dim 40).
constexpr std::size_t kBlockSize = 32u;

constexpr std::array<std::size_t, 3> kChunkSizes = {1024u, 128u, 1u};

// The most logits rows a chunk produces. pa::plan_chunks ends a chunk early
// rather than sample more.
constexpr std::size_t kMaxSampled = 16u;

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

}  // anonymous namespace

void ov::npuw::pa::check_static_contract(const std::shared_ptr<ov::Model>& model) {
    const auto reject = [](const std::string& why) {
        OPENVINO_THROW("PACompiledModel: the model is outside the flat-token PA contract: ", why);
    };

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
            reject("unsupported input '" + name + "'");
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
            reject(std::string("no '") + required + "' input");
        }
    }
    for (const char* name : {"input_ids", "position_ids"}) {
        const auto& rank = model->input(name).get_partial_shape().rank();
        if (rank.is_dynamic() || rank.get_length() != 1) {
            reject(std::string("'") + name + "' is not a 1-D token stream");
        }
    }
    const auto& outputs = model->outputs();
    if (outputs.size() != 1 || outputs.front().get_any_name() != "logits") {
        reject("expected one 'logits' output");
    }
    const auto& lshape = outputs.front().get_partial_shape();
    if (lshape.rank().is_dynamic() || lshape.rank().get_length() != 3 || lshape[1].is_dynamic() ||
        lshape[2].is_dynamic()) {
        reject("'logits' is not [rows, 1, vocab]");
    }
}

std::shared_ptr<ov::Model> ov::npuw::pa::derive_static_variant(const std::shared_ptr<ov::Model>& model,
                                                               std::size_t chunk_size,
                                                               std::size_t sampled_dim) {
    auto derived = model->clone();
    derived->reshape({{"input_ids", ov::PartialShape{static_cast<int64_t>(chunk_size)}},
                      {"position_ids", ov::PartialShape{static_cast<int64_t>(chunk_size)}},
                      {"sampled_tokens_indices", ov::PartialShape{static_cast<int64_t>(sampled_dim)}}});

    const auto i32 = [](int32_t value) {
        return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {value});
    };
    auto n_rows = std::make_shared<ov::op::v8::Gather>(derived->input("subsequence_begins"), i32(-1), i32(0));
    auto rows = std::make_shared<ov::op::v4::Range>(i32(0), n_rows, i32(1), ov::element::i32);

    // One zeros constant per output geometry, shared by the layers.
    std::map<std::pair<ov::element::Type, ov::Shape>, std::shared_ptr<ov::Node>> zeros;
    for (const auto& node : derived->get_ordered_ops()) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }
        for (std::size_t i = 0; i < 3; ++i) {  // query, key, value
            pa->input(i).replace_source_output(std::make_shared<ov::op::v8::Gather>(pa->input_value(i), rows, i32(0)));
        }
        pa->validate_and_infer_types();

        const auto& attn_shape = pa->get_output_partial_shape(0);
        OPENVINO_ASSERT(
            attn_shape.rank().is_static() && attn_shape.rank().get_length() == 2 && attn_shape[1].is_static(),
            "PA: unexpected PagedAttention output shape ",
            attn_shape);
        const auto type = pa->get_output_element_type(0);
        const ov::Shape shape{chunk_size, static_cast<std::size_t>(attn_shape[1].get_length())};
        auto& zero = zeros[{type, shape}];
        if (!zero) {
            zero = ov::op::v0::Constant::create(type, shape, std::vector<float>{0.f});
        }

        const auto consumers = pa->output(0).get_target_inputs();
        auto padded = std::make_shared<ov::op::v3::ScatterUpdate>(zero, rows, pa->output(0), i32(0));
        for (auto consumer : consumers) {
            consumer.replace_source_output(padded);
        }
    }
    derived->validate_nodes_and_infer_types();
    derived->set_friendly_name(model->get_friendly_name() + "_pa_token_" + std::to_string(chunk_size));
    return derived;
}

namespace {

ov::npuw::PACompiledModel::Variants compile_static_variants(const std::shared_ptr<ov::Model>& model,
                                                            const std::shared_ptr<const ov::IPlugin>& plugin,
                                                            const ov::AnyMap& properties) {
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
    ov::npuw::pa::check_static_contract(model);

    // NPU* keys configure this plugin and DEVICE_ID names an NPU device; the
    // executing device would reject both. Everything else is forwarded.
    ov::AnyMap inner_config;
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU") || key == ov::device::id.name()) {
            continue;
        }
        inner_config.emplace(key, value);
    }

    ov::npuw::PACompiledModel::Variants variants;
    for (const auto chunk_size : kChunkSizes) {
        const auto sampled_dim = std::min(chunk_size, kMaxSampled);
        auto derived = ov::npuw::pa::derive_static_variant(model, chunk_size, sampled_dim);
        auto compiled = plugin->get_core()->compile_model(derived, device, inner_config);
        OPENVINO_ASSERT(compiled != nullptr,
                        "PA static variant chunk_size=",
                        chunk_size,
                        " failed to compile on ",
                        device);
        LOG_INFO("PA: compiled static variant chunk_size=" << chunk_size << " on " << device);
        variants.emplace(chunk_size, std::move(compiled));
    }
    return variants;
}

// The model with the KV cache element types and shapes the device resolved,
// which the pipeline reads off the ports. They are the same in every variant.
std::shared_ptr<ov::Model> with_cache_geometry(const std::shared_ptr<ov::Model>& model,
                                               const ov::SoPtr<ov::ICompiledModel>& variant) {
    std::unordered_map<std::string, ov::Output<const ov::Node>> resolved;
    for (const auto& input : variant->inputs()) {
        resolved.emplace(input.get_any_name(), input);
    }
    auto stamped = model->clone();
    for (const auto& param : stamped->get_parameters()) {
        const auto& name = param->get_output_tensor(0).get_any_name();
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
            const auto& port = resolved.at(name);
            param->set_element_type(port.get_element_type());
            param->set_partial_shape(port.get_partial_shape());
        }
    }
    stamped->validate_nodes_and_infer_types();
    return stamped;
}

}  // anonymous namespace

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : PACompiledModel(compile_static_variants(model, plugin, properties), model, plugin) {}

ov::npuw::PACompiledModel::PACompiledModel(Variants&& variants,
                                           const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(with_cache_geometry(model, variants.begin()->second), plugin),
      m_variants(std::move(variants)) {}

void ov::npuw::PACompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not support export_model()");
}

std::shared_ptr<const ov::Model> ov::npuw::PACompiledModel::get_runtime_model() const {
    return m_variants.begin()->second->get_runtime_model();
}

void ov::npuw::PACompiledModel::set_property(const ov::AnyMap& properties) {
    // A clear error instead of the executing device's "unsupported property".
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU")) {
            OPENVINO_THROW("PACompiledModel: '", key, "' cannot be changed after the model is compiled");
        }
    }
    for (auto& [chunk_size, variant] : m_variants) {
        variant->set_property(properties);
    }
}

ov::Any ov::npuw::PACompiledModel::get_property(const std::string& name) const {
    // Everything but NPUW_PA is the executing device's, including
    // execution_devices, which the CB pipeline reads to pick its block size.
    // The variants share the device and the config, so the largest answers.
    const auto& executing = m_variants.begin()->second;
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    if (name == ov::supported_properties.name()) {
        auto props = executing->get_property(name).as<std::vector<ov::PropertyName>>();
        props.emplace_back(std::string(::intel_npu::NPUW_PA::key()), ov::PropertyMutability::RO);
        return props;
    }
    return executing->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::PACompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const ov::ICompiledModel>(shared_from_this());
    return std::make_shared<PAInferRequest>(self, m_variants);
}

ov::npuw::PAInferRequest::PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                         const PACompiledModel::Variants& variants)
    : ov::ISyncInferRequest(compiled_model) {
    // Dynamic ports start empty; the pipeline sets or reshapes them per dispatch.
    const auto allocate = [&](const ov::Output<const ov::Node>& port) {
        allocate_tensor(port, [&](ov::SoPtr<ov::ITensor>& tensor) {
            const auto& shape = port.get_partial_shape();
            tensor = ov::get_tensor_impl(
                ov::Tensor(port.get_element_type(), shape.is_static() ? shape.to_shape() : shape.get_min_shape()));
        });
    };
    for (const auto& input : get_inputs()) {
        m_inputs_by_name.emplace(input.get_any_name(), input);
        allocate(input);
    }
    for (const auto& output : get_outputs()) {
        allocate(output);
    }

    for (const auto& [chunk_size, compiled] : variants) {
        ChunkRequest chunk;
        chunk.request = compiled->create_infer_request();
        OPENVINO_ASSERT(chunk.request != nullptr, "PA chunk model requires a valid infer request");
        for (const auto& input : compiled->inputs()) {
            chunk.inputs.emplace(input.get_any_name(), input);
        }
        chunk.logits = compiled->outputs().front();
        const auto bind = [&](const char* name) {
            const auto& port = chunk.inputs.at(name);
            auto tensor = ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape()));
            chunk.request->set_tensor(port, tensor);
            return tensor;
        };
        chunk.input_ids = bind("input_ids");
        chunk.position_ids = bind("position_ids");
        chunk.sampled_tokens_indices = bind("sampled_tokens_indices");
        m_chunk_requests.emplace(chunk_size, std::move(chunk));
        m_chunk_sizes.push_back(chunk_size);
    }
}

ov::npuw::pa::Dispatch ov::npuw::PAInferRequest::parse_dispatch() const {
    // The static contract guarantees every input read here.
    const auto get = [&](const char* name) {
        return get_tensor(m_inputs_by_name.at(name));
    };

    pa::Dispatch d;
    d.past_lens = read_vec<int32_t>(get("past_lens"));
    d.subsequence_begins = read_vec<int32_t>(get("subsequence_begins"));
    const auto mcl_vec = read_vec<int32_t>(get("max_context_len"));
    OPENVINO_ASSERT(!mcl_vec.empty(), "PA dispatch: max_context_len is not set");
    d.max_context_len = mcl_vec.front();
    d.input_ids_size = static_cast<int64_t>(get("input_ids")->get_size());
    d.position_ids_token_count = static_cast<int64_t>(get("position_ids")->get_size());
    d.has_block_table = true;
    d.block_indices = read_vec<int32_t>(get("block_indices"));
    d.block_indices_begins = read_vec<int32_t>(get("block_indices_begins"));
    d.has_sampled_tokens = true;
    d.sampled_tokens_indices = read_vec<int64_t>(get("sampled_tokens_indices"));
    return d;
}

void ov::npuw::PAInferRequest::run_chunk(ChunkRequest& chunk, const pa::Dispatch& d, const pa::Chunk& c) {
    const auto caller = [&](const std::string& name) {
        return get_tensor(m_inputs_by_name.at(name));
    };
    const auto set = [&](const char* name, const ov::SoPtr<ov::ITensor>& tensor) {
        chunk.request->set_tensor(chunk.inputs.at(name), tensor);
    };

    // The chunk's rows at the front, zeros in the padding rows.
    for (const auto& [name, dst] :
         {std::pair{"input_ids", chunk.input_ids}, std::pair{"position_ids", chunk.position_ids}}) {
        const auto src = caller(name);
        const auto row_bytes = dst->get_element_type().size();
        const auto chunk_bytes = static_cast<std::size_t>(c.tokens) * row_bytes;
        auto* out = static_cast<uint8_t*>(dst->data());
        std::memcpy(out,
                    static_cast<const uint8_t*>(src->data()) + static_cast<std::size_t>(c.begin) * row_bytes,
                    chunk_bytes);
        std::memset(out + chunk_bytes, 0, dst->get_byte_size() - chunk_bytes);
    }

    const auto controls = pa::chunk_controls(d, c);
    set("past_lens", make_vec_tensor(controls.past_lens));
    set("subsequence_begins", make_vec_tensor(controls.subsequence_begins));
    set("block_indices", make_vec_tensor(controls.block_indices));
    set("block_indices_begins", make_vec_tensor(controls.block_indices_begins));
    set("max_context_len", caller("max_context_len"));
    // Only the scores output reads the window, and the contract has none.
    if (chunk.inputs.count("score_aggregation_window") > 0) {
        set("score_aggregation_window", make_vec_tensor(std::vector<int32_t>(controls.past_lens.size(), 0)));
    }
    for (const auto& [name, port] : chunk.inputs) {
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
            chunk.request->set_tensor(port, caller(name));
        }
    }

    // Padded with row 0, whose logits are dropped.
    const auto& local = controls.sampled_tokens_indices;
    auto& gather = chunk.sampled_tokens_indices;
    OPENVINO_ASSERT(local.size() <= gather->get_size(),
                    "PA chunk samples ",
                    local.size(),
                    " rows, its variant produces ",
                    gather->get_size());
    std::fill(std::copy(local.begin(), local.end(), gather->data<int64_t>()),
              gather->data<int64_t>() + gather->get_size(),
              int64_t{0});

    chunk.request->infer();

    const auto out = chunk.request->get_tensor(chunk.logits);
    const auto logits = get_tensor(get_outputs().front());
    const auto& shape = logits->get_shape();
    const auto row_bytes = shape.at(1) * shape.at(2) * logits->get_element_type().size();
    const auto* src = static_cast<const uint8_t*>(out->data());
    auto* dst = static_cast<uint8_t*>(logits->data());
    for (std::size_t j = 0; j < controls.logits_rows.size(); ++j) {
        std::memcpy(dst + controls.logits_rows[j] * row_bytes, src + j * row_bytes, row_bytes);
    }
}

void ov::npuw::PAInferRequest::log_dispatch_io(bool outputs) const {
    if (ov::npuw::get_log_level() < ov::npuw::LogLevel::Verbose) {
        return;
    }
    LOG_VERB("PA dispatch #" << m_dispatch_idx << (outputs ? " outputs:" : " inputs:"));
    LOG_BLOCK();
    for (const auto& port : outputs ? get_outputs() : get_inputs()) {
        LOG_VERB(port.get_any_name() << ": " << ov::npuw::util::TensorBrief{get_tensor(port)});
    }
}

void ov::npuw::PAInferRequest::infer() {
    log_dispatch_io(/*outputs=*/false);
    const auto dispatch = parse_dispatch();
    pa::validate_dispatch(dispatch, kBlockSize, m_dispatch_idx);
    const auto chunks = pa::plan_chunks(dispatch, m_chunk_sizes, kMaxSampled);
    if (ov::npuw::get_log_level() >= ov::npuw::LogLevel::Verbose) {
        std::ostringstream plan;
        for (const auto& c : chunks) {
            plan << (&c == &chunks.front() ? "" : "+") << c.tokens;
            if (static_cast<std::size_t>(c.tokens) != c.chunk_size) {
                plan << "/" << c.chunk_size;
            }
        }
        LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << dispatch.sequences() << " subsequence(s), "
                                 << dispatch.tokens() << " token(s), " << dispatch.sampled_tokens_indices.size()
                                 << " sampled, chunks " << plan.str());
    }

    // One row per sampled token, in the caller's order.
    const auto logits = get_tensor(get_outputs().front());
    auto shape = logits->get_shape();
    shape.at(0) = dispatch.sampled_tokens_indices.size();
    logits->set_shape(shape);

    // Chunks run in flat order, so a subsequence's earlier rows are in the cache
    // before its later rows attend to them. That matches one infer because no
    // subsequence reads a block another one writes in the same dispatch (GenAI
    // forks blocks copy-on-write and caches only computed prefixes).
    for (const auto& c : chunks) {
        run_chunk(m_chunk_requests.at(c.chunk_size), dispatch, c);
    }
    log_dispatch_io(/*outputs=*/true);
    ++m_dispatch_idx;
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::PAInferRequest::query_state() const {
    // Stateless: the KV cache is the caller's.
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::PAInferRequest::get_profiling_info() const {
    return {};
}
