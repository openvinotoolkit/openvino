#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_registry.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include <cstdio>
#include <limits>

namespace ov::intel_gpu::ocl::moe {

LookaheadContext::LookaheadContext(cldnn::network& network)
    : m_network(network), m_compute_stream(network.get_stream_ptr()) {
    build_once(network);
    m_prefetcher = std::make_unique<ExpertPrefetcher>(network.get_engine(), network.get_config(), m_compute_stream);
}

LookaheadContext::~LookaheadContext() {
    for (const auto& item : m_recall) {
        std::fprintf(stderr, "LES_PREDICTOR_RECALL layer=%zu matched=%llu selections=%llu\n", item.first,
                     static_cast<unsigned long long>(item.second.matched), static_cast<unsigned long long>(item.second.selections));
    }
    m_prefetcher.reset();
    try {
        m_compute_stream->finish();
    } catch (const std::exception& exception) {
        GPU_DEBUG_INFO << "Lookahead compute teardown failed: " << exception.what() << std::endl;
    }
}

void LookaheadContext::pause_for_prefill() {
    m_compute_stream->flush();
    m_prefetcher->advance(++m_epoch, std::numeric_limits<uint32_t>::max());
    m_prefetcher->drain();
    m_cursor.reset();
}

void LookaheadContext::build_once(cldnn::network& network) {
    std::call_once(m_init_flag, [&] {
        for (const auto& id : network.get_input_ids()) {
            if (id == "parameter:past_lens") {
                m_past_lens_input = id;
                break;
            }
        }
        m_links = network.get_program()->get_les_router_links();
        for (const auto& id : network.get_executed_primitive_ids()) {
            auto instance = network.get_primitive(id);
            if (instance->type() == cldnn::moe_3gemm_fused_compressed::type_id()) {
                m_layer_indices.emplace(id, static_cast<uint32_t>(m_layer_indices.size()));
                LookaheadLayerDesc descriptor;
                descriptor.moe_inst = instance.get();
                m_layers.push_back(std::move(descriptor));
            }
        }
        GPU_DEBUG_INFO << "LES: " << m_layer_indices.size() << " MoE layers, " << m_links.size() << " router links" << std::endl;
        for (const auto& link : m_links) {
            GPU_DEBUG_INFO << "LES link " << link.chain.source_moe << " -> " << link.chain.target_moe << " chain="
                           << static_cast<int>(link.chain.status) << " router=" << static_cast<int>(link.status) << " " << link.rejection_reason
                           << std::endl;
        }
        const auto& predictor = network.get_config().get_les_predictor();
        std::optional<LesPredictorModel> predictor_model;
        if (predictor) {
            // The predictor replaces the next-layer router, so only the plain chain topology is required.
            predictor_model = load_les_predictor_model(predictor);
            for (const auto& id : network.get_input_ids()) {
                if (id == "parameter:les_prev_hidden" && network.get_program()->get_node(id).get_users().empty()) {
                    m_previous_input = id;
                    for (const auto& head : predictor_model->heads) {
                        m_previous_rows.emplace(head.first, m_previous_rows.size());
                    }
                    break;
                }
            }
            m_links.clear();
            for (const auto& chain : network.get_program()->get_les_moe_links()) {
                const auto target = m_layer_indices.find(chain.target_moe);
                if (chain.status != cldnn::program::les_link_status::candidate || target == m_layer_indices.end() ||
                    !m_layer_indices.count(chain.source_moe)) {
                    continue;
                }
                const auto layer = les_layer_of(chain.target_moe).value_or(predictor_model->heads.begin()->first + target->second);
                const auto head = predictor_model->heads.find(layer);
                if (head == predictor_model->heads.end()) {
                    continue;
                }
                cldnn::program::les_router_link link;
                link.chain = chain;
                link.status = cldnn::program::les_router_status::supported;
                link.experts = head->second.experts;
                link.top_k = predictor_model->top_k;
                link.sigmoid_bias = true;
                m_links.push_back(std::move(link));
                m_predictor_layers.emplace(chain.target_moe, layer);
            }
        }
        for (size_t index = 0; index < m_links.size(); ++index) {
            const auto& link = m_links[index];
            if (link.status != cldnn::program::les_router_status::supported ||
                !m_layer_indices.count(link.chain.source_moe) || !m_layer_indices.count(link.chain.target_moe)) {
                continue;
            }
            m_source_links.emplace(link.chain.source_moe, index);
            const auto target_index = m_layer_indices.at(link.chain.target_moe);
            auto& descriptor = m_layers[target_index];
            if (!predictor_model) {
                descriptor.router_fc_inst = network.get_primitive(link.projection).get();
                descriptor.weight_mem = network.get_primitive(link.weights)->output_memory_ptr();
            }
            descriptor.num_experts = static_cast<uint32_t>(link.experts);
            const auto shape = network.get_program()->get_node(link.chain.target_hidden).get_output_layout().get_partial_shape();
            if (shape.rank().is_static() && shape.size() > 0 && shape[shape.size() - 1].is_static()) {
                descriptor.hidden_size = static_cast<uint32_t>(shape[shape.size() - 1].get_length());
            }
            descriptor.is_supported = true;
            m_moe_to_next_index.emplace(network.get_primitive(link.chain.source_moe).get(), target_index);
            if (!predictor_model) {
                m_routers.emplace(link.chain.target_moe,
                                  std::make_unique<LookaheadRouter>(network, link.chain.target_hidden, link.router, link.projection));
            }
        }
        if (predictor_model) {
            for (const auto& [id, layer] : m_predictor_layers) {
                const auto& head = predictor_model->heads.at(layer);
                const auto& target = m_layers[m_layer_indices.at(id)];
                OPENVINO_ASSERT(head.experts == target.num_experts && head.hidden == target.hidden_size,
                                "LES predictor head of layer ", layer, " does not match the model: experts ", head.experts, " vs ",
                                target.num_experts, ", hidden ", head.hidden, " vs ", target.hidden_size);
            }
            m_predictor = std::make_unique<LesPredictor>(network, *predictor_model);
            if (!m_previous_input.empty()) {
                const ov::PartialShape shape{static_cast<int64_t>(m_previous_rows.size()), 1,
                                             static_cast<int64_t>(predictor_model->heads.begin()->second.hidden)};
                cldnn::topology topology;
                topology.add(cldnn::input_layout("state", {shape, cldnn::data_types::f32, cldnn::format::bfyx}));
                topology.add(cldnn::reorder("packed", cldnn::input_info("state"), {shape, cldnn::data_types::f16, cldnn::format::bfyx}));
                auto program = cldnn::program::build_program(network.get_engine(), topology, network.get_config(), true);
                m_previous_converter = cldnn::network::allocate_network(network.get_stream_ptr(), program, true, true);
            }
            GPU_DEBUG_INFO << "LES predictor: " << m_predictor->head_count() << " heads, " << m_predictor_layers.size()
                           << " MoE layers mapped, top_k " << m_predictor->top_k() << std::endl;
        }
    });
}

const LookaheadLayerDesc* LookaheadContext::get_next_layer(const cldnn::primitive_inst* current_moe) const {
    const auto found = m_moe_to_next_index.find(current_moe);
    if (found == m_moe_to_next_index.end()) {
        return nullptr;
    }
    const auto& descriptor = m_layers[found->second];
    return descriptor.is_supported ? &descriptor : nullptr;
}

bool LookaheadContext::prediction_enabled(const cldnn::primitive_id& id, size_t tokens) {
    if (m_past_lens_input.empty()) {
        return tokens == 1 && !m_previous_converter;
    }
    const auto layer = m_layer_indices.at(id);
    if (!m_last_checked_layer || layer <= *m_last_checked_layer) {
        const auto past = m_network.get_primitive(m_past_lens_input)->output_memory_ptr();
        m_has_past = false;
        if (past && past->get_layout().count() == 1 && past->get_layout().data_type == cldnn::data_types::i32) {
            cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> lengths(past, *m_compute_stream);
            m_has_past = lengths[0] > 0;
        }
    }
    m_last_checked_layer = layer;
    if (!m_has_past || !m_previous_converter) {
        return m_has_past && (tokens == 1 || m_predictor != nullptr);
    }
    const auto input = m_network.get_primitive(m_previous_input)->output_memory_ptr();
    return input && input->get_layout().get_shape().at(1) == 1;
}

void LookaheadContext::enter(const cldnn::primitive_id& id) {
    m_prefetcher->rethrow_if_failed();
    auto found = m_layer_indices.find(id);
    if (found == m_layer_indices.end()) {
        return;
    }
    if (!m_cursor || found->second <= *m_cursor) {
        ++m_epoch;
        m_prefetcher->advance(m_epoch, 0);
    }
    m_cursor = found->second;
}

void LookaheadContext::demand_partitioned(const cldnn::primitive_id& id) {
    auto found = m_layer_indices.find(id);
    if (found != m_layer_indices.end()) {
        m_prefetcher->advance(m_epoch, found->second);
    }
}

std::shared_ptr<IExpertWeightProvider> LookaheadContext::bind(const cldnn::primitive_id& id,
                                                            std::shared_ptr<IExpertWeightProvider> provider) {
    const auto existing = m_providers.find(id);
    if (existing != m_providers.end()) {
        return existing->second;
    }
    auto offload = std::dynamic_pointer_cast<OffloadExpertWeightProvider>(provider);
    if (offload) {
        m_prefetcher->bind(offload);
        m_providers[id] = std::move(offload);
    }
    return provider;
}

void LookaheadContext::predict(const LookaheadLayerDesc& next_layer,
                               const cldnn::memory::ptr& hidden,
                               const cldnn::event::ptr& ready) {
    if (!next_layer.is_supported || !next_layer.moe_inst) {
        return;
    }
    const auto& target = next_layer.moe_inst->id();
    auto& state = m_predictions[target];
    if (state.consumed && !state.consumed->load(std::memory_order_acquire)) {
        return;
    }
    auto result = [&] {
        if (auto* head = predictor_head(target)) {
            auto& stream = m_network.get_stream();
            stream.enqueue_barrier({ready});
            const size_t bytes = size_t{head->hidden} * sizeof(ov::float16);
            const size_t tokens = hidden->get_layout().count() / head->hidden;
            if (m_previous_converter) {
                if (m_previous_epoch != m_epoch) {
                    const auto input = m_network.get_primitive(m_previous_input)->output_memory_ptr();
                    const ov::Shape expected{m_previous_rows.size(), 1, head->hidden};
                    OPENVINO_ASSERT(input && input->get_layout().get_shape() == expected &&
                                    input->get_layout().data_type == cldnn::data_types::f32,
                                    "Native LES feedback expects [exported layers, one sequence, hidden] FP32");
                    m_previous_converter->set_input_data("state", input);
                    auto converted = m_previous_converter->execute({ready}).at("packed");
                    m_previous_values = converted.get_memory(false);
                    m_previous_ready = converted.get_event();
                    m_previous_epoch = m_epoch;
                }
                if (m_previous_ready) {
                    stream.enqueue_barrier({m_previous_ready});
                }
                const auto row = m_previous_rows.at(m_predictor_layers.at(target));
                head->previous->copy_from(stream, *m_previous_values, row * bytes, 0, bytes, false);
            }
            return head->execute(hidden, stream.enqueue_marker({}, true), tokens * m_predictor->top_k());
        }
        return m_routers.at(target)->enqueue(hidden, ready);
    }();
    state = {hidden, ready, result.indices, result.weights, result.ready,
             std::make_shared<std::atomic<bool>>(true), false, result.cpu_indices, m_epoch};
}

void LookaheadContext::record_recall(const cldnn::primitive_id& id, const std::vector<uint32_t>& actual) {
    const auto layer = m_predictor_layers.find(id);
    const auto prediction = m_predictions.find(id);
    if (layer == m_predictor_layers.end() || prediction == m_predictions.end() || prediction->second.epoch != m_epoch ||
        !prediction->second.cpu_indices) {
        return;
    }
    prediction->second.prediction_ready->wait();
    const auto& predicted = *prediction->second.cpu_indices;
    auto& recall = m_recall[layer->second];
    recall.selections += actual.size();
    const size_t predicted_k = m_predictor->top_k();
    const size_t tokens = predicted.size() / predicted_k;
    OPENVINO_ASSERT(tokens && actual.size() % tokens == 0, "Invalid LES recall row count");
    const size_t actual_k = actual.size() / tokens;
    for (size_t index = 0; index < actual.size(); ++index) {
        const auto first = predicted.begin() + index / actual_k * predicted_k;
        recall.matched += std::find(first, first + predicted_k, actual[index]) != first + predicted_k;
    }
}

LesPredictor::Head* LookaheadContext::predictor_head(const cldnn::primitive_id& id) {
    const auto layer = m_predictor_layers.find(id);
    return m_predictor && layer != m_predictor_layers.end() ? m_predictor->find(layer->second) : nullptr;
}

void LookaheadContext::update_previous(const cldnn::primitive_id& id,
                                       const cldnn::memory::ptr& hidden,
                                       const std::vector<cldnn::event::ptr>& ready) {
    auto* head = predictor_head(id);
    if (m_previous_converter || !head || !hidden) {
        return;
    }
    const auto& layout = hidden->get_layout();
    const size_t row_bytes = size_t{head->hidden} * sizeof(ov::float16);
    if (layout.data_type != cldnn::data_types::f16 || layout.format != cldnn::format::bfyx || layout.get_linear_size() != layout.count() ||
        layout.count() < head->hidden || layout.count() % head->hidden != 0) {
        GPU_DEBUG_INFO << "LES predictor: cannot keep the previous-token input of " << id << " for layout " << layout.to_short_string() << std::endl;
        return;
    }
    std::vector<cldnn::event::ptr> dependencies;
    for (const auto& event : ready) {
        if (event) {
            dependencies.push_back(event);
        }
    }
    auto& stream = m_network.get_stream();
    stream.enqueue_barrier(dependencies);
    // The last row is the newest token of a prefill chunk.
    head->previous->copy_from(stream, *hidden, layout.count() * sizeof(ov::float16) - row_bytes, 0, row_bytes, false);
}

void LookaheadContext::prefetch(const cldnn::primitive_inst* current_moe) {
    const auto* next_layer = get_next_layer(current_moe);
    if (!next_layer) {
        return;
    }
    const auto& target = next_layer->moe_inst->id();
    auto& state = m_predictions[target];
    auto provider = m_providers.find(target);
    if (state.submitted || !state.prediction_ready || provider == m_providers.end()) {
        return;
    }
    auto weights = provider->second;
    const auto& link = m_links.at(m_source_links.at(current_moe->id()));
    if (!weights || weights->resident_capacity() <= link.top_k) {
        return;
    }
    state.submitted = true;
    state.consumed->store(false, std::memory_order_release);
    m_network.get_stream().flush();
    m_prefetcher->submit_prefetch({m_layer_indices.at(target), m_epoch,
                                  state.predicted_indices, state.prediction_ready, weights, state.consumed, state.cpu_indices});
}

std::shared_ptr<LookaheadContext> LookaheadRegistry::acquire(cldnn::network& network) {
    const auto& config = network.get_config();
    if (!config.get_moe_lookahead_prefetch() || config.get_offload_ratio() == 0 || network.get_id() == 0) {
        return {};
    }
    OPENVINO_ASSERT(config.get_offload_ratio() < 100 && config.get_num_streams() == 1 && !config.get_record_replay(),
                    "LES requires offload ratio in [1,99], one stream and no record/replay");
    static std::mutex mutex;
    static std::unordered_map<uint32_t, std::weak_ptr<LookaheadContext>> registry;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto entry = registry.begin(); entry != registry.end();) {
        entry = entry->second.expired() ? registry.erase(entry) : std::next(entry);
    }
    auto& entry = registry[network.get_id()];
    auto context = entry.lock();
    if (!context) {
        context = std::make_shared<LookaheadContext>(network);
        entry = context;
    }
    return context;
}

}