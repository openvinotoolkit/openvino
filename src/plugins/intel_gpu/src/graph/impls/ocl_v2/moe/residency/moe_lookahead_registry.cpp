#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_registry.hpp"
#include <limits>

namespace ov::intel_gpu::ocl::moe {

LookaheadContext::LookaheadContext(cldnn::network& network)
    : m_network(network), m_compute_stream(network.get_stream_ptr()) {
    build_once(network);
    m_prefetcher = std::make_unique<ExpertPrefetcher>(network.get_engine(), network.get_config(), m_compute_stream);
}

LookaheadContext::~LookaheadContext() {
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
        const auto& predictor_path = network.get_config().get_les_predictor();
        std::optional<LesPredictorModel> predictor_model;
        if (!predictor_path.empty()) {
            // The predictor replaces the next-layer router, so only the plain chain topology is required.
            predictor_model = load_les_predictor_model(predictor_path);
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
            const auto copied = head->input->copy_from(stream, *hidden, 0, 0, size_t{head->hidden} * sizeof(ov::float16), false);
            return head->router->enqueue(head->input, copied);
        }
        return m_routers.at(target)->enqueue(hidden, ready);
    }();
    state = {hidden, ready, result.indices, result.weights, result.ready,
             std::make_shared<std::atomic<bool>>(true), false, result.cpu_indices};
}

LesPredictor::Head* LookaheadContext::predictor_head(const cldnn::primitive_id& id) {
    const auto layer = m_predictor_layers.find(id);
    return m_predictor && layer != m_predictor_layers.end() ? m_predictor->find(layer->second) : nullptr;
}

void LookaheadContext::update_previous(const cldnn::primitive_id& id,
                                       const cldnn::memory::ptr& hidden,
                                       const std::vector<cldnn::event::ptr>& ready) {
    auto* head = predictor_head(id);
    if (!head || !hidden) {
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
    head->input->copy_from(stream, *hidden, layout.count() * sizeof(ov::float16) - row_bytes, row_bytes, row_bytes, false);
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