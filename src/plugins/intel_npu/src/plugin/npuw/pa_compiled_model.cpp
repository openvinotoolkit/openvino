// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_compiled_model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "util.hpp"

namespace {

// The token sizes the static variants are derived for. 16 serves the decode
// batch (several single-token subsequences in one infer), 1 the single
// sequence generation case, the two larger ones the prefill chunks.
const std::vector<std::size_t> kVariantTokenDims = {1024u, 128u, 16u, 1u};

// The most sampled-token rows a variant produces: one per subsequence of a
// decode batch, or the validated tokens of one prefill chunk.
constexpr std::size_t kMaxSampled = 16u;

// The KV cache block size on the PA device. The CPU PagedAttention executor
// requires 32, and the CB pipeline assumes 32 for any device that is not a
// GPU. The compiled cache ports may carry a larger dim (a quantized cache
// keeps its scales in the block), so the block size is not read off them.
constexpr std::size_t kBlockSize = 32u;

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

}  // anonymous namespace

void ov::npuw::PACompiledModel::require_static_contract(const std::shared_ptr<ov::Model>& model) {
    static const std::unordered_set<std::string> known = {"input_ids",
                                                          "position_ids",
                                                          "past_lens",
                                                          "subsequence_begins",
                                                          "block_indices",
                                                          "block_indices_begins",
                                                          "max_context_len",
                                                          "score_aggregation_window",
                                                          "sampled_tokens_indices"};
    const auto reject = [](const std::string& why) {
        OPENVINO_THROW("PACompiledModel: the model is outside the static PA contract: ", why);
    };
    std::unordered_set<std::string> seen;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        if (ov::npuw::util::is_pa_kv_cache_name(name)) {
            continue;
        }
        if (known.count(name) == 0) {
            reject("unsupported input '" + name + "'");
        }
        seen.insert(name);
    }
    for (const char* required :
         {"input_ids", "position_ids", "past_lens", "subsequence_begins", "block_indices", "sampled_tokens_indices"}) {
        if (seen.count(required) == 0) {
            reject(std::string("missing input '") + required + "'");
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
        reject("expected a single 'logits' output");
    }
    const auto& lshape = outputs.front().get_partial_shape();
    if (lshape.rank().is_dynamic() || lshape.rank().get_length() != 3 || !lshape[1].is_static() ||
        !lshape[2].is_static()) {
        reject("'logits' is not [rows, 1, vocab] with static row geometry");
    }
}

namespace {

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

// A fresh vector tensor with the port's (integer) element type.
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

// A zeroed 1-D tensor of the port's element type.
ov::SoPtr<ov::ITensor> make_zeroed(const ov::Output<const ov::Node>& port, std::size_t n) {
    auto tensor = ov::get_tensor_impl(ov::Tensor(port.get_element_type(), ov::Shape{n}));
    std::memset(tensor->data(), 0, tensor->get_byte_size());
    return tensor;
}

// n elements of a 1-D tensor, from src[src_start] to dst[dst_start].
void copy_1d(const ov::SoPtr<ov::ITensor>& src,
             int64_t src_start,
             const ov::SoPtr<ov::ITensor>& dst,
             int64_t dst_start,
             int64_t n) {
    const auto esize = src->get_element_type().size();
    std::memcpy(static_cast<uint8_t*>(dst->data()) + static_cast<std::size_t>(dst_start) * esize,
                static_cast<const uint8_t*>(src->data()) + static_cast<std::size_t>(src_start) * esize,
                static_cast<std::size_t>(n) * esize);
}

}  // anonymous namespace

std::shared_ptr<ov::Model> ov::npuw::PACompiledModel::derive_static_variant(
    const std::shared_ptr<ov::Model>& base_model,
    std::size_t token_dim,
    std::size_t sampled_dim) {
    auto derived = base_model->clone();
    derived->reshape({{"input_ids", ov::PartialShape{static_cast<int64_t>(token_dim)}},
                      {"position_ids", ov::PartialShape{static_cast<int64_t>(token_dim)}},
                      {"sampled_tokens_indices", ov::PartialShape{static_cast<int64_t>(sampled_dim)}}});

    const auto begins = derived->input("subsequence_begins");
    const auto i32_scalar = [](int32_t v) {
        return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {v});
    };
    auto n_tokens = std::make_shared<ov::op::v8::Gather>(begins, i32_scalar(-1), i32_scalar(0));
    auto rows = std::make_shared<ov::op::v4::Range>(i32_scalar(0), n_tokens, i32_scalar(1), ov::element::i32);
    const auto axis0 = i32_scalar(0);

    // The zeros the islands scatter into: one constant per distinct output
    // geometry, shared by all layers (a per-layer copy would be hundreds of
    // megabytes for the largest variant).
    std::map<std::pair<ov::element::Type, ov::Shape>, std::shared_ptr<ov::Node>> zeros_by_geometry;
    const auto zeros_for = [&](const ov::element::Type& type, const ov::Shape& shape) {
        auto& zeros = zeros_by_geometry[{type, shape}];
        if (!zeros) {
            zeros = ov::op::v0::Constant::create(type, shape, std::vector<float>{0.f});
        }
        return zeros;
    };

    for (const auto& node : derived->get_ordered_ops()) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }
        for (std::size_t i = 0; i < 3; ++i) {  // query, key, value
            auto sliced = std::make_shared<ov::op::v8::Gather>(pa->input_value(i), rows, axis0);
            pa->input(i).replace_source_output(sliced);
        }
        pa->validate_and_infer_types();
        const auto& out_shape = pa->get_output_partial_shape(0);
        OPENVINO_ASSERT(out_shape.rank().is_static() && out_shape.rank().get_length() == 2 && out_shape[1].is_static(),
                        "PA: unexpected PagedAttention output shape ",
                        out_shape);
        const auto zeros = zeros_for(pa->get_output_element_type(0),
                                     ov::Shape{token_dim, static_cast<std::size_t>(out_shape[1].get_length())});
        const auto consumers = pa->output(0).get_target_inputs();
        auto padded = std::make_shared<ov::op::v3::ScatterUpdate>(zeros, rows, pa->output(0), axis0);
        for (auto consumer : consumers) {
            consumer.replace_source_output(padded);
        }
    }
    derived->validate_nodes_and_infer_types();
    derived->set_friendly_name(base_model->get_friendly_name() + "_pa_token_" + std::to_string(token_dim));
    return derived;
}

ov::npuw::PACompiledModel::Prepared ov::npuw::PACompiledModel::prepare(const std::shared_ptr<ov::Model>& model,
                                                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                       const ov::AnyMap& properties) {
    // The fallback device is an internal development knob, not a config
    // option - an env var keeps it out of user configs (and blob cache keys).
    // Only CPU is supported for now: a GPU device would also need the remote
    // context forwarded for the pipeline's cache allocation.
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    const std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    OPENVINO_ASSERT(device == "CPU",
                    "The PagedAttention fallback device is CPU for now, got OPENVINO_NPUW_PA_DEVICE=",
                    device);

    require_static_contract(model);

    // NPUW_*, NPU_USE_NPUW and NPU_* keys are this plugin's configuration and
    // must not reach the executing device (which would reject them as
    // unsupported); everything else (e.g. KV_CACHE_PRECISION, performance
    // hints) is the executing device's business and is forwarded. DEVICE_ID
    // names an NPU device (e.g. NPU.3600), so it stays behind as well.
    ov::AnyMap inner_config;
    for (const auto& [key, value] : properties) {
        if (ov::npuw::util::starts_with(key, "NPU") || key == ov::device::id.name()) {
            continue;
        }
        inner_config.emplace(key, value);
    }

    Prepared prepared;
    for (const auto token_dim : kVariantTokenDims) {
        const auto sampled_dim = std::min(token_dim, kMaxSampled);
        auto derived = derive_static_variant(model, token_dim, sampled_dim);
        auto compiled = plugin->get_core()->compile_model(derived, device, inner_config);
        OPENVINO_ASSERT(compiled != nullptr, "PA: variant token_dim=", token_dim, " failed to compile on ", device);
        LOG_INFO("PA: compiled static variant token_dim=" << token_dim << " sampled_dim=" << sampled_dim << " on "
                                                          << device);
        prepared.variants.emplace(token_dim, std::move(compiled));
    }

    // The exposed model is the dynamic one with the KV cache geometry the
    // device fixed at compile time stamped onto its cache ports: element
    // types and block shapes are what the pipeline allocates its pools from.
    // Identical across variants and layers by construction.
    prepared.model = model->clone();
    prepared.block_size = kBlockSize;
    std::unordered_map<std::string, ov::Output<const ov::Node>> reference_ports;
    for (const auto& input : prepared.variants.begin()->second->inputs()) {
        reference_ports.emplace(input.get_any_name(), input);
    }
    for (const auto& param : prepared.model->get_parameters()) {
        const auto& name = param->get_output_tensor(0).get_any_name();
        if (!ov::npuw::util::is_pa_kv_cache_name(name)) {
            continue;
        }
        const auto port_it = reference_ports.find(name);
        OPENVINO_ASSERT(port_it != reference_ports.end(), "PA: the compiled variant has no '", name, "' input");
        const auto& port = port_it->second;
        param->set_element_type(port.get_element_type());
        param->set_partial_shape(port.get_partial_shape());
        param->validate_and_infer_types();
        // [num_blocks (dyn), kv_heads, block dim, head dim]; the block dim
        // holds at least the block's tokens.
        const auto& shape = port.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static() &&
                            shape[2].get_length() >= static_cast<int64_t>(kBlockSize),
                        "PA: unexpected KV cache geometry ",
                        shape,
                        " for '",
                        name,
                        "' on ",
                        device);
    }
    prepared.model->validate_nodes_and_infer_types();
    LOG_INFO("PA: " << prepared.variants.size() << " static variant(s) on " << device << ", KV block size "
                    << prepared.block_size);
    return prepared;
}

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : PACompiledModel(prepare(model, plugin, properties), plugin) {}

ov::npuw::PACompiledModel::PACompiledModel(Prepared&& prepared, const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_model(std::move(prepared.model)),
      m_variants(std::move(prepared.variants)),
      m_block_size(prepared.block_size) {}

void ov::npuw::PACompiledModel::export_model(std::ostream&) const {
    OPENVINO_THROW_NOT_IMPLEMENTED("PACompiledModel does not support export_model()");
}

std::shared_ptr<const ov::Model> ov::npuw::PACompiledModel::get_runtime_model() const {
    return m_model;
}

void ov::npuw::PACompiledModel::set_property(const ov::AnyMap& properties) {
    // Everything is fixed at compile time; a clear error beats a silent no-op.
    OPENVINO_ASSERT(properties.empty(),
                    "PACompiledModel: '",
                    properties.begin()->first,
                    "' cannot be changed after the model is compiled");
}

ov::Any ov::npuw::PACompiledModel::get_property(const std::string& name) const {
    if (name == std::string(::intel_npu::NPUW_PA::key())) {
        return true;
    }
    // The CB pipeline picks its block size off the execution device.
    if (name == ov::execution_devices.name()) {
        return std::vector<std::string>{get_plugin()->get_device_name()};
    }
    if (name == ov::model_name.name()) {
        return m_model->get_friendly_name();
    }
    if (name == ov::optimal_number_of_infer_requests.name()) {
        return static_cast<uint32_t>(1u);
    }
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName(std::string(::intel_npu::NPUW_PA::key()), ov::PropertyMutability::RO),
            ov::PropertyName(ov::execution_devices.name(), ov::PropertyMutability::RO),
            ov::PropertyName(ov::model_name.name(), ov::PropertyMutability::RO),
            ov::PropertyName(ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO)};
    }
    OPENVINO_THROW("PACompiledModel: unsupported property ", name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::PACompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const ov::ICompiledModel>(shared_from_this());
    return std::make_shared<PAInferRequest>(self, m_block_size, m_variants);
}

ov::npuw::PAInferRequest::PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                         std::size_t block_size,
                                         const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants)
    : ov::ISyncInferRequest(compiled_model),
      m_block_size(block_size) {
    // The caller's tensors live here. Dynamic ports start with their minimal
    // shape; the pipeline resizes or replaces them per dispatch.
    for (const auto& input : get_inputs()) {
        m_inputs_by_name.emplace(input.get_any_name(), input);
        const auto& pshape = input.get_partial_shape();
        allocate_tensor(input, [&](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::get_tensor_impl(
                ov::Tensor(input.get_element_type(), pshape.is_static() ? pshape.to_shape() : pshape.get_min_shape()));
        });
    }
    const auto& logits = get_outputs().front();
    m_logits_node = logits.get_node();
    const auto& lshape = logits.get_partial_shape();
    m_logits = ov::get_tensor_impl(ov::Tensor(logits.get_element_type(),
                                              ov::Shape{0u,
                                                        static_cast<std::size_t>(lshape[1].get_length()),
                                                        static_cast<std::size_t>(lshape[2].get_length())}));

    for (const auto& [token_dim, compiled] : variants) {
        VariantRequest v;
        v.request = compiled->create_infer_request();
        OPENVINO_ASSERT(v.request != nullptr, "PA variant requires a valid infer request");
        for (const auto& input : compiled->inputs()) {
            v.inputs.emplace(input.get_any_name(), input);
        }
        v.logits = compiled->outputs().front();
        v.token_dim = token_dim;
        v.sampled_dim = static_cast<std::size_t>(v.inputs.at("sampled_tokens_indices").get_shape().at(0));
        v.input_ids = make_zeroed(v.inputs.at("input_ids"), v.token_dim);
        v.position_ids = make_zeroed(v.inputs.at("position_ids"), v.token_dim);
        v.sampled_tokens_indices = make_zeroed(v.inputs.at("sampled_tokens_indices"), v.sampled_dim);
        v.request->set_tensor(v.inputs.at("input_ids"), v.input_ids);
        v.request->set_tensor(v.inputs.at("position_ids"), v.position_ids);
        v.request->set_tensor(v.inputs.at("sampled_tokens_indices"), v.sampled_tokens_indices);
        m_max_sampled = std::max(m_max_sampled, v.sampled_dim);
        m_variants.emplace(token_dim, std::move(v));
        m_variant_token_dims.push_back(token_dim);
    }
}

ov::npuw::pa::Dispatch ov::npuw::PAInferRequest::parse_dispatch() const {
    const auto get = [&](const char* name) {
        auto it = m_inputs_by_name.find(name);
        OPENVINO_ASSERT(it != m_inputs_by_name.end(), "PA model has no '", name, "' input");
        return get_tensor(it->second);
    };

    pa::Dispatch d;
    d.past_lens = as_i64_vec(get("past_lens"));
    d.subsequence_begins = as_i64_vec(get("subsequence_begins"));
    const auto mcl_vec = as_i64_vec(get("max_context_len"));
    OPENVINO_ASSERT(!mcl_vec.empty(), "PA dispatch: max_context_len is not set");
    d.max_context_len = mcl_vec.front();
    d.input_ids_size = static_cast<int64_t>(get("input_ids")->get_size());
    const auto& pos_shape = get("position_ids")->get_shape();
    OPENVINO_ASSERT(!pos_shape.empty(), "PA dispatch: position_ids has no shape");
    d.position_ids_token_count = static_cast<int64_t>(pos_shape.back());
    d.block_indices = as_i64_vec(get("block_indices"));
    d.block_indices_begins = as_i64_vec(get("block_indices_begins"));
    d.sampled_tokens_indices = as_i64_vec(get("sampled_tokens_indices"));
    return d;
}

void ov::npuw::PAInferRequest::run_chunk(VariantRequest& v, const pa::Dispatch& d, const pa::Chunk& chunk) {
    const auto set = [&](const char* name, const ov::SoPtr<ov::ITensor>& tensor) {
        auto it = v.inputs.find(name);
        OPENVINO_ASSERT(it != v.inputs.end(), "PA variant has no '", name, "' input");
        v.request->set_tensor(it->second, tensor);
    };
    const auto caller = [&](const char* name) {
        return get_tensor(m_inputs_by_name.at(name));
    };

    // Token streams: the pieces' slices packed at the front, zeros behind.
    for (const auto& [name, dst] : {std::pair{"input_ids", v.input_ids}, std::pair{"position_ids", v.position_ids}}) {
        const auto src = caller(name);
        std::memset(dst->data(), 0, dst->get_byte_size());
        int64_t local = 0;
        for (const auto& p : chunk.pieces) {
            copy_1d(src, d.subsequence_begins[p.seq] + p.offset, dst, local, p.tokens);
            local += p.tokens;
        }
    }

    // Controls: one subsequence per piece, rebased past the tokens the piece
    // skips, each with its subsequence's full block table. The context stays
    // the caller's; nothing is written beyond the real tokens.
    std::vector<int64_t> past, blocks, sub{0}, bib{0};
    for (const auto& p : chunk.pieces) {
        past.push_back(d.past_lens[p.seq] + p.offset);
        sub.push_back(sub.back() + p.tokens);
        const auto b0 = d.block_indices_begins[p.seq], b1 = d.block_indices_begins[p.seq + 1];
        blocks.insert(blocks.end(), d.block_indices.begin() + b0, d.block_indices.begin() + b1);
        bib.push_back(bib.back() + (b1 - b0));
    }
    set("past_lens", make_ctrl_tensor(v.inputs.at("past_lens"), past));
    set("subsequence_begins", make_ctrl_tensor(v.inputs.at("subsequence_begins"), sub));
    set("block_indices", make_ctrl_tensor(v.inputs.at("block_indices"), blocks));
    set("block_indices_begins", make_ctrl_tensor(v.inputs.at("block_indices_begins"), bib));
    set("max_context_len", caller("max_context_len"));
    // Score aggregation only feeds the (unused) scores output; a zero window
    // per piece disables it, so the caller's tensor is not consulted.
    if (const auto it = v.inputs.find("score_aggregation_window"); it != v.inputs.end()) {
        v.request->set_tensor(it->second, make_zeroed(it->second, chunk.pieces.size()));
    }

    // Sampled rows falling into this chunk, remembered with their position in
    // the caller's sampled_tokens_indices order; the gather is padded with
    // row 0, whose logits are discarded.
    std::vector<int64_t> local_sti;
    std::vector<std::size_t> out_rows;
    int64_t local_base = 0;
    for (const auto& p : chunk.pieces) {
        const auto g0 = d.subsequence_begins[p.seq] + p.offset;
        for (std::size_t i = 0; i < d.sampled_tokens_indices.size(); ++i) {
            const auto g = d.sampled_tokens_indices[i];
            if (g >= g0 && g < g0 + p.tokens) {
                local_sti.push_back(local_base + (g - g0));
                out_rows.push_back(i);
            }
        }
        local_base += p.tokens;
    }
    OPENVINO_ASSERT(local_sti.size() <= v.sampled_dim,
                    "PA dispatch: a chunk samples ",
                    local_sti.size(),
                    " tokens, the variant produces ",
                    v.sampled_dim);
    local_sti.resize(v.sampled_dim, 0);
    const auto sti_type = v.sampled_tokens_indices->get_element_type();
    OPENVINO_ASSERT(sti_type == ov::element::i64 || sti_type == ov::element::i32, "PA: unexpected gather index type");
    if (sti_type == ov::element::i64) {
        std::copy(local_sti.begin(), local_sti.end(), v.sampled_tokens_indices->data<int64_t>());
    } else {
        std::transform(local_sti.begin(), local_sti.end(), v.sampled_tokens_indices->data<int32_t>(), [](int64_t x) {
            return static_cast<int32_t>(x);
        });
    }

    v.request->infer();

    const auto out = v.request->get_tensor(v.logits);
    const auto& oshape = m_logits->get_shape();
    const auto row_bytes = oshape.at(1) * oshape.at(2) * m_logits->get_element_type().size();
    const auto* src = static_cast<const uint8_t*>(out->data());
    auto* dst = static_cast<uint8_t*>(m_logits->data());
    for (std::size_t j = 0; j < out_rows.size(); ++j) {
        std::memcpy(dst + out_rows[j] * row_bytes, src + j * row_bytes, row_bytes);
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
        LOG_VERB(name << ": " << tensor_brief(get_tensor(port), !ov::npuw::util::is_pa_kv_cache_name(name)));
    }
}

void ov::npuw::PAInferRequest::infer() {
    log_dispatch_io(/*outputs=*/false);
    const auto d = parse_dispatch();
    pa::validate_dispatch(d, m_block_size, m_dispatch_idx);
    const auto chunks = pa::plan_dispatch(d, m_variant_token_dims, m_max_sampled);

    // One logits row per sampled token, in the caller's order.
    const auto& oshape = m_logits->get_shape();
    m_logits = ov::get_tensor_impl(
        ov::Tensor(m_logits->get_element_type(), ov::Shape{d.sampled_tokens_indices.size(), oshape[1], oshape[2]}));

    LOG_VERB("PA dispatch #" << m_dispatch_idx << ": " << d.sequences() << " subsequence(s), " << d.tokens()
                             << " token(s), " << d.sampled_tokens_indices.size()
                             << " sampled; chunks tokens/variant: " << pa::to_string(chunks));
    for (const auto& chunk : chunks) {
        run_chunk(m_variants.at(chunk.token_dim), d, chunk);
    }
    log_dispatch_io(/*outputs=*/true);
    ++m_dispatch_idx;
}

ov::SoPtr<ov::ITensor> ov::npuw::PAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    if (port.get_node() == m_logits_node) {
        return m_logits;
    }
    return ov::ISyncInferRequest::get_tensor(port);
}

void ov::npuw::PAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                          const ov::SoPtr<ov::ITensor>& tensor) {
    ov::ISyncInferRequest::set_tensor(port, tensor);
    // The KV cache pools are shared with every variant as they come; the
    // other inputs are read per dispatch.
    const auto& name = port.get_any_name();
    if (ov::npuw::util::is_pa_kv_cache_name(name)) {
        for (auto& [token_dim, v] : m_variants) {
            v.request->set_tensor(v.inputs.at(name), tensor);
        }
    }
}

void ov::npuw::PAInferRequest::check_tensors() const {
    // The logits output is produced here, not held in the base storage, and
    // the inputs are validated against the PA contract in infer().
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::PAInferRequest::query_state() const {
    return {};
}

std::vector<ov::ProfilingInfo> ov::npuw::PAInferRequest::get_profiling_info() const {
    return {};
}
