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
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_npu/config/npuw.hpp"
#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "partitioning/patterns/opt.hpp"
#include "util.hpp"

namespace {

// GPU, or one of several GPUs such as GPU.1.
bool is_gpu(const std::string& device) {
    return ov::npuw::util::starts_with(device, "GPU");
}

// The block size of the CPU cache, as the CB pipeline fixes it, for the
// dispatch check. A GPU has its own (16, or 256 with XAttention), so its
// dispatches skip the block table coverage check.
constexpr std::size_t kBlockSize = 32u;

constexpr std::array<std::size_t, 5> kChunkSizes = {1024u, 128u, 8u, 4u, 1u};

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
    // A flat token stream, which SDPAToPagedAttention unsqueezes for the layers.
    for (const char* name : {"input_ids", "position_ids"}) {
        const auto input = model->input(name);
        const auto& rank = input.get_partial_shape().rank();
        if (rank.is_dynamic() || rank.get_length() != 1) {
            return false;
        }
        for (const auto& reader : input.get_target_inputs()) {
            if (!ov::is_type<ov::op::v0::Unsqueeze>(reader.get_node())) {
                return false;
            }
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

// The target of a Reshape to `shape`, whose one dynamic dimension is the token count.
std::shared_ptr<ov::Node> reshape_target(const ov::PartialShape& shape) {
    std::vector<int64_t> target;
    for (const auto& dim : shape) {
        target.push_back(dim.is_static() ? dim.get_length() : -1);
    }
    OPENVINO_ASSERT(std::count(target.begin(), target.end(), -1) == 1, "PA: no single token dimension in ", shape);
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target.size()}, target);
}

std::shared_ptr<ov::Model> derive_pa_semi_static_model(const std::shared_ptr<ov::Model>& base_model,
                                                       std::size_t chunk_size) {
    auto derived = base_model->clone();
    derived->reshape({{"input_ids", ov::PartialShape{1, static_cast<int64_t>(chunk_size)}},
                      {"position_ids", ov::PartialShape{1, static_cast<int64_t>(chunk_size)}}});
    for (const auto& node : derived->get_ops()) {
        OPENVINO_ASSERT(
            !ov::is_type<ov::op::PagedAttentionExtension>(node) || node->get_input_shape(0)[0] == chunk_size,
            "PA: the layers before ",
            node->get_friendly_name(),
            " do not keep the token count");
    }
    derived->set_friendly_name(base_model->get_friendly_name() + "_pa_token_" + std::to_string(chunk_size));
    return derived;
}

// The variants go through NPUW on this plugin's device, partitioned by
// make_variant_config().
std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> compile_pa_semi_static_variants(
    const std::shared_ptr<ov::Model>& base_model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& variant_config) {
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> variants;
    const auto prepared = ov::npuw::PACompiledModel::prepare_variant_model(base_model);
    for (const auto chunk_size : kChunkSizes) {
        auto derived = derive_pa_semi_static_model(prepared, chunk_size);
        auto compiled = plugin->get_core()->compile_model(derived, plugin->get_device_name(), variant_config);
        OPENVINO_ASSERT(compiled != nullptr, "PA semi-static derivation failed to compile chunk_size=", chunk_size);
        LOG_INFO("PA: compiled semi-static variant chunk_size=" << chunk_size << " through NPUW");
        variants.emplace(chunk_size, std::move(compiled));
    }

    return variants;
}

}  // anonymous namespace

std::shared_ptr<ov::Model> ov::npuw::PACompiledModel::prepare_variant_model(
    const std::shared_ptr<ov::Model>& base_model) {
    for (const char* name : {"input_ids", "position_ids"}) {
        const auto& rank = base_model->input(name).get_partial_shape().rank();
        OPENVINO_ASSERT(rank.is_static() && rank.get_length() == 1, "PA: '", name, "' is not a 1-D token stream");
    }
    auto model = base_model->clone();

    // The inputs that stay token-first, with the shapes they have for it.
    std::vector<std::pair<ov::Input<ov::Node>, ov::PartialShape>> token_first;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::PagedAttentionExtension>(node)) {
            continue;
        }
        // Queries, keys and values: [tokens, heads * head_size].
        for (std::size_t i = 0; i < 3; ++i) {
            token_first.emplace_back(node->input(i), node->get_input_partial_shape(i));
        }
        // The result returns to the layers through a Reshape to
        // [tokens, 1, heads, head_size], now to [1, tokens, heads, head_size].
        for (const auto& reader : node->output(0).get_target_inputs()) {
            auto shape = reader.get_node()->get_output_partial_shape(0);
            OPENVINO_ASSERT(ov::is_type<ov::op::v1::Reshape>(reader.get_node()) && shape.size() == 4 && shape[1] == 1,
                            "PA: unexpected reader of a PA op: ",
                            reader.get_node()->get_friendly_name());
            std::swap(shape[0], shape[1]);
            reader.get_node()->input(1).replace_source_output(reshape_target(shape));
        }
    }
    // The Gather of the sampled rows: [tokens, 1, hidden].
    for (const auto& reader : model->input("sampled_tokens_indices").get_target_inputs()) {
        OPENVINO_ASSERT(ov::is_type<ov::op::v8::Gather>(reader.get_node()), "PA: the sampled rows are not gathered");
        token_first.emplace_back(reader.get_node()->input(0), reader.get_node()->get_input_partial_shape(0));
    }

    for (const char* name : {"input_ids", "position_ids"}) {
        const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(model->input(name).get_node_shared_ptr());
        param->set_partial_shape(ov::PartialShape{1, -1});
        for (const auto& unsqueeze : param->output(0).get_target_inputs()) {
            for (const auto& reader : unsqueeze.get_node()->output(0).get_target_inputs()) {
                reader.replace_source_output(param->output(0));
            }
        }
    }
    for (const auto& [input, shape] : token_first) {
        input.replace_source_output(
            std::make_shared<ov::op::v1::Reshape>(input.get_source_output(), reshape_target(shape), false));
    }
    model->validate_nodes_and_infer_types();

    // Tied embeddings decompress one table for the embedding Gather and the LM
    // head, which would make the table a link between their subgraphs. The
    // gather lift has the Gather read the weights itself, so the decompression
    // is the LM head's alone. NPUW runs it on its own only when every subgraph
    // is a function, which the variants cannot ask for.
    ov::pass::Manager manager("PA variants");
    const auto lift = manager.register_pass<ov::pass::GraphRewrite>();
    lift->add_matcher<ov::npuw::patterns::opt::DQLiftGatherAsymCW>();
    lift->add_matcher<ov::npuw::patterns::opt::DQLiftGatherSymCW>();
    lift->add_matcher<ov::npuw::patterns::opt::DQLiftGatherSymGQ>();
    lift->add_matcher<ov::npuw::patterns::opt::DQLiftGatherCW>();
    manager.run_passes(model);
    return model;
}

ov::AnyMap ov::npuw::PACompiledModel::make_variant_config(const ov::AnyMap& properties,
                                                          const std::string& pa_device,
                                                          const std::string& weights_bank,
                                                          const std::optional<util::NPUDesc>& npudesc) {
    // The variants start from the NPU defaults of the NPUW LLM path. On top of
    // them every PA op is a block of its own, kept off NPU, and the static
    // repeated blocks around them fold onto NPU and share one weights bank
    // across the variants, which are clones of one model. The PA model picks
    // its logits rows itself, so there is no last token to slice out, and the
    // tail that does it has a dynamic row count, so it cannot be made a
    // function like the static subgraphs. The caller's properties override
    // these defaults, and extend the compilation mode parameters.
    auto config = util::get_default_common_config(npudesc);
    config.erase(std::string(::intel_npu::NPUW_SLICE_OUT::key()));
    config.erase(std::string(::intel_npu::NPUW_FUNCALL_FOR_ALL::key()));
    // The tail picks a dynamic number of logits rows, which the NPU compiler
    // cannot take, so it goes to the PA device.
    config[std::string(::intel_npu::NPUW_DEVICES::key())] = "NPU," + pa_device;
    if (is_gpu(pa_device)) {
        // GPU workaround. At f16 the GPU drops the convert F16IC puts on an
        // f16 input and adds the residual of the tail's first FC in place into
        // the input buffer, which crashes. With f32 links there is no such input.
        config[std::string(::intel_npu::NPUW_F16IC::key())] = std::string("NO");
    }
    config[std::string(::intel_npu::NPUW_ONLINE_PIPELINE::key())] = std::string("REP");
    config[std::string(::intel_npu::NPUW_ONLINE_ISOLATE::key())] = std::string("Op:PagedAttentionExtension/pa");
    config[std::string(::intel_npu::NPUW_ONLINE_AVOID::key())] = std::string("Op:PagedAttentionExtension/NPU");
    config[std::string(::intel_npu::NPUW_ONLINE_KEEP_BLOCKS_TAGGED::key())] = std::string("pa");
    const auto compilation_mode_params = std::string(::intel_npu::COMPILATION_MODE_PARAMS::key());
    for (const auto& [key, value] : properties) {
        if (key == compilation_mode_params) {
            config[key] =
                ov::npuw::util::merge_compilation_mode_params(config[key].as<std::string>(), value.as<std::string>());
        } else {
            config[key] = value;
        }
    }

    // A variant is an ordinary NPUW model: it must not select the PA front-end
    // again, and a partitioning chosen for the whole model does not apply to it.
    config.erase(std::string(::intel_npu::NPUW_PA::key()));
    config.erase(std::string(::intel_npu::NPUW_PLAN::key()));
    config[std::string(::intel_npu::NPUW_SUBMODEL_DEVICE::key())] = "last:" + pa_device;
    config[ov::intel_npu::use_npuw.name()] = std::string("YES");
    config[std::string(::intel_npu::NPUW_WEIGHTS_BANK::key())] = weights_bank;
    return config;
}

ov::npuw::PACompiledModel::DynamicModel ov::npuw::PACompiledModel::compile_dynamic(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    // An env var, not an option, so it stays out of user configs.
    const char* device_env = std::getenv("OPENVINO_NPUW_PA_DEVICE");
    const std::string device = (device_env != nullptr && device_env[0] != '\0') ? device_env : "CPU";
    OPENVINO_ASSERT(device == "CPU" || is_gpu(device),
                    "The PagedAttention device is CPU or a GPU, got OPENVINO_NPUW_PA_DEVICE=",
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
    if (is_gpu(device)) {
        // GPU workaround. With its compressed cache, the default for a model
        // with 4-bit weights, a GPU gives NaN from a PA op of a fixed token
        // count, as the variants have them. An f16 cache does not, so the GPU
        // gets one unless the caller asks for a precision.
        inner_config.emplace(ov::hint::kv_cache_precision.name(), ov::element::f16);
    }

    LOG_INFO("PA: compiling the dynamic PA model 1:1 on " << device);
    auto compiled = plugin->get_core()->compile_model(model, device, inner_config);
    OPENVINO_ASSERT(compiled != nullptr, "PACompiledModel requires a valid inner compiled model");
    return {device, std::move(compiled)};
}

ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties)
    : PACompiledModel(model, plugin, properties, compile_dynamic(model, plugin, properties)) {}

// The ports are the inner model's, see inputs(). The CB pipeline allocates the
// cache pools from this model's context when the execution device is a GPU, so
// with PA on GPU the context is the dynamic model's. The CPU has none.
ov::npuw::PACompiledModel::PACompiledModel(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<const ov::IPlugin>& plugin,
                                           const ov::AnyMap& properties,
                                           DynamicModel dynamic)
    : ov::npuw::ICompiledModel(
          nullptr,
          plugin,
          is_gpu(dynamic.device) ? dynamic.compiled->get_context() : ov::SoPtr<ov::IRemoteContext>{}),
      m_compiled_model(std::move(dynamic.compiled)),
      m_block_size(is_gpu(dynamic.device) ? 0u : kBlockSize) {
    const auto& device = dynamic.device;
    if (is_chunkable_pa_model(model)) {
        // One bank for all the variants, unless the caller names one.
        const auto bank_it = properties.find(std::string(::intel_npu::NPUW_WEIGHTS_BANK::key()));
        const auto caller_bank = bank_it == properties.end() ? std::string{} : bank_it->second.as<std::string>();
        const auto bank = caller_bank.empty() ? ov::npuw::util::generate_random_string() : caller_bank;
        auto variant_config =
            make_variant_config(properties, device, bank, util::extract_npu_descriptor(plugin, properties));
        // NPUW builds the PA blocks of the variants as new models, without the
        // rt_info runtime options the CPU may take its cache settings from and
        // without the 4-bit weights the GPU picks a u4 cache for. Both read the
        // same cache pools, so the variants take the dynamic model's effective
        // settings.
        const auto cache_keys = is_gpu(device) ? std::vector<std::string>{ov::hint::kv_cache_precision.name()}
                                               : std::vector<std::string>{ov::key_cache_precision.name(),
                                                                          ov::value_cache_precision.name(),
                                                                          ov::key_cache_group_size.name(),
                                                                          ov::value_cache_group_size.name()};
        for (const auto& key : cache_keys) {
            variant_config[key] = m_compiled_model->get_property(key);
        }
        m_semi_static_models = compile_pa_semi_static_variants(model, plugin, variant_config);
    } else {
        LOG_INFO("PA: model is outside the chunkable flat-token contract; every dispatch runs 1:1");
    }
    LOG_INFO("PA: " << m_semi_static_models.size() << " semi-static variant(s), PA ops on " << device);
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
    return std::make_shared<PAInferRequest>(self, std::move(inner_request), m_semi_static_models, m_block_size);
}

ov::npuw::PAInferRequest::PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                                         ov::SoPtr<ov::IAsyncInferRequest> inner_request,
                                         const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants,
                                         std::size_t block_size)
    : ov::ISyncInferRequest(compiled_model),
      m_inner_request(std::move(inner_request)),
      m_block_size(block_size) {
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
                                         int64_t global_start,
                                         int64_t n_chunk_tokens) {
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

    // A variant takes its tokens as [1, tokens], the dynamic model as [tokens].
    const auto tokens = [&](const char* name) {
        const auto view = slice(name, global_start, n_chunk_tokens);
        if (chunk.inputs.at(name).get_partial_shape().size() == 1) {
            return view;
        }
        const ov::Shape shape = {1, static_cast<std::size_t>(n_chunk_tokens)};
        return ov::get_tensor_impl(ov::Tensor(view->get_element_type(), shape, view->data()));
    };
    set("input_ids", tokens("input_ids"));
    set("position_ids", tokens("position_ids"));

    // The subsequences the chunk's tokens belong to, each past its tokens
    // before the chunk, with its full block table.
    const auto& sub = d.subsequence_begins;
    const auto first = static_cast<int32_t>(global_start);
    const auto last = static_cast<int32_t>(global_start + n_chunk_tokens);
    const auto seq = std::upper_bound(sub.begin(), sub.end(), first) - sub.begin() - 1;
    const auto blocks_begin = d.block_indices_begins[seq];
    std::vector<int32_t> past_lens;
    std::vector<int32_t> subsequence_begins = {0};
    std::vector<int32_t> block_indices_begins = {0};
    auto seq_end = seq;
    for (; sub[seq_end] < last; ++seq_end) {
        past_lens.push_back(d.past_lens[seq_end] + std::max(first - sub[seq_end], 0));
        subsequence_begins.push_back(std::min(sub[seq_end + 1], last) - first);
        block_indices_begins.push_back(d.block_indices_begins[seq_end + 1] - blocks_begin);
    }
    set("past_lens", make_vec_tensor(past_lens));
    set("subsequence_begins", make_vec_tensor(subsequence_begins));
    set("block_indices", slice("block_indices", blocks_begin, block_indices_begins.back()));
    set("block_indices_begins", make_vec_tensor(block_indices_begins));
    if (m_inputs_by_name.count("score_aggregation_window") > 0) {
        set("score_aggregation_window", slice("score_aggregation_window", seq, seq_end - seq));
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

    const auto& sub = d.subsequence_begins;
    const auto is_decode = [&](int64_t seq) {
        return sub[seq + 1] - sub[seq] == 1;
    };
    for (int64_t s = 0, end = 0; s < d.sequences(); s = end) {
        // A sequence, or a decode batch: one-token sequences in a row.
        end = s + 1;
        while (is_decode(s) && end < d.sequences() && is_decode(end)) {
            ++end;
        }
        const auto seq_len = sub[end] - sub[s];
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
            run_chunk(chunk, d, sub[s] + off, n);
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
    pa::validate_dispatch(dispatch, m_block_size, m_dispatch_idx);
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
