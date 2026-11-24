#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_registry.hpp"
#include <limits>

namespace ov::intel_gpu::ocl::moe {

LookaheadContext::LookaheadContext(cldnn::network& network)
    : m_network(network), m_compute_stream(network.get_stream_ptr()) {
    build_once(network);
    m_prefetcher = std::make_unique<ExpertPrefetcher>(network.get_engine(), network.get_config());
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
        for (size_t index = 0; index < m_links.size(); ++index) {
            const auto& link = m_links[index];
            if (link.status != cldnn::program::les_router_status::supported ||
                !m_layer_indices.count(link.chain.source_moe) || !m_layer_indices.count(link.chain.target_moe)) {
                continue;
            }
            m_source_links.emplace(link.chain.source_moe, index);
            const auto target_index = m_layer_indices.at(link.chain.target_moe);
            auto& descriptor = m_layers[target_index];
            descriptor.router_fc_inst = network.get_primitive(link.projection).get();
            descriptor.weight_mem = network.get_primitive(link.weights)->output_memory_ptr();
            descriptor.num_experts = static_cast<uint32_t>(link.experts);
            const auto shape = network.get_program()->get_node(link.chain.target_hidden).get_output_layout().get_partial_shape();
            if (shape.rank().is_static() && shape.size() > 0 && shape[shape.size() - 1].is_static()) {
                descriptor.hidden_size = static_cast<uint32_t>(shape[shape.size() - 1].get_length());
            }
            descriptor.is_supported = true;
            m_moe_to_next_index.emplace(network.get_primitive(link.chain.source_moe).get(), target_index);
            m_routers.emplace(link.chain.target_moe,
                              std::make_unique<LookaheadRouter>(network, link.chain.target_hidden, link.router));
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
    auto result = m_routers.at(target)->enqueue(hidden, ready);
    state = {hidden, ready, result.indices, result.weights, result.ready,
             std::make_shared<std::atomic<bool>>(true), false};
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
                                  state.predicted_indices, state.prediction_ready, weights, state.consumed});
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