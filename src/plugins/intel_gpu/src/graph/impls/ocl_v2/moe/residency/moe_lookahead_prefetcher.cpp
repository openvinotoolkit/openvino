#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_prefetcher.hpp"

#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>

#include "../expert_weight_providers.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl::moe {
namespace {

class ReadyEvent final : public cldnn::ocl::ocl_event {
public:
    explicit ReadyEvent(const cl::Context& context) : ocl_event(cl::UserEvent(context)) {}
    ~ReadyEvent() override {
        if (!m_terminal.exchange(true)) {
            clSetUserEventStatus(get().get(), CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
    }

    void signal(cl_int status) {
        if (!m_terminal.exchange(true)) {
            OPENVINO_ASSERT(clSetUserEventStatus(get().get(), status) == CL_SUCCESS,
                            "Cannot signal lookahead completion event");
        }
    }

private:
    std::atomic<bool> m_terminal{false};
};

cl_event native_event(const cldnn::event::ptr& event) {
    auto native = std::dynamic_pointer_cast<cldnn::ocl::ocl_base_event>(event);
    OPENVINO_ASSERT(native && native->get().get(), "Lookahead requires a native OpenCL event");
    return native->get().get();
}

cl_int event_status(const cldnn::event::ptr& event) {
    if (!event) {
        return CL_COMPLETE;
    }
    cl_int status = CL_QUEUED;
    OPENVINO_ASSERT(clGetEventInfo(native_event(event), CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof(status), &status, nullptr) == CL_SUCCESS,
                    "Cannot query lookahead event status");
    return status;
}

void signal(const cldnn::event::ptr& event, cl_int status) {
    auto ready = std::dynamic_pointer_cast<ReadyEvent>(event);
    OPENVINO_ASSERT(ready, "Lookahead slot does not own a completion bridge");
    ready->signal(status);
}

void consume(PrefetchTask& task) {
    if (task.consumed) {
        task.consumed->store(true, std::memory_order_release);
    }
}

}

struct ExpertPrefetcher::State {
    struct Wakeup {
        std::mutex mutex;
        std::condition_variable cv;
        std::mutex notification_mutex;
        std::condition_variable notification_cv;
        uint64_t revision = 0;

        void notify() {
            {
                std::lock_guard<std::mutex> lock(notification_mutex);
                ++revision;
            }
            notification_cv.notify_all();
            cv.notify_all();
        }
    };
    struct Binding {
        std::shared_ptr<LRUCache> cache;
        cldnn::MOECompressed::Config config;
        std::vector<size_t> offsets;
        cldnn::moe_weights weights;
        std::unique_ptr<moe_otd::ParallelWeightReader> reader;
    };
    struct Batch {
        size_t remaining;
        cldnn::event::ptr ready;
    };
    struct Job {
        std::shared_ptr<Binding> binding;
        LRUCache::Load load;
        std::shared_ptr<Batch> batch;
        uint64_t generation = 0;
        uint32_t layer = 0;
    };

    explicit State(cldnn::engine& engine, const cldnn::ExecutionConfig& config)
        : dma(moe_otd::create_copy_stream(engine, config)),
          context(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context()) {
        for (auto& chunk : staging.chunks) {
            chunk.memory = engine.allocate_memory(
                cldnn::layout{cldnn::data_types::u8, cldnn::format::bfyx,
                              {1, 1, 1, static_cast<int32_t>(moe_otd::LookaheadStagingPool::chunk_bytes)}},
                cldnn::allocation_type::usm_host, false);
        }
        worker = std::thread([this] { run(); });
    }

    ~State() {
        {
            std::lock_guard<std::mutex> lock(wakeup->mutex);
            stop = true;
        }
        wakeup->notify();
        worker.join();
        try {
            dma->finish();
        } catch (const std::exception& exception) {
            GPU_DEBUG_INFO << "Lookahead DMA teardown failed: " << exception.what() << std::endl;
        }
    }

    void watch(const cldnn::event::ptr& event) {
        if (!event) {
            return;
        }
        auto callback_data = std::make_unique<std::shared_ptr<Wakeup>>(wakeup);
        auto* callback_ptr = callback_data.release();
        const auto result = clSetEventCallback(native_event(event), CL_COMPLETE,
            [](cl_event, cl_int, void* data) {
                std::unique_ptr<std::shared_ptr<Wakeup>> notification(static_cast<std::shared_ptr<Wakeup>*>(data));
                auto target = *notification;
                target->notify();
            }, callback_ptr);
        if (result != CL_SUCCESS) {
            delete callback_ptr;
            OPENVINO_THROW("Cannot subscribe to lookahead event: ", result);
        }
    }

    void finish(Job& job, cl_int status) {
        if (status == CL_COMPLETE) {
            job.binding->cache->complete_load(job.load);
        }
        signal(job.load.ready_event, status);
        if (job.batch && (status != CL_COMPLETE || --job.batch->remaining == 0)) {
            signal(job.batch->ready, status);
        }
    }

    void discard_locked() {
        for (auto& task : predictions) {
            consume(task);
        }
        predictions.clear();
        for (auto& job : demand) {
            abandoned.push_back(std::move(job));
        }
        demand.clear();
        for (auto& job : speculative) {
            job.binding->cache->cancel_load(job.load);
            abandoned.push_back(std::move(job));
        }
        speculative.clear();
        std::lock_guard<std::mutex> completion_lock(completion_mutex);
        completion_closed = true;
        for (const auto& weak : ready_events) {
            if (auto event = weak.lock()) {
                abandoned_events.push_back(std::move(event));
            }
        }
        ready_events.clear();
    }

    void fail_abandoned() {
        std::vector<Job> jobs;
        jobs.swap(abandoned);
        for (auto& job : jobs) {
            finish(job, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
        std::vector<cldnn::event::ptr> events;
        events.swap(abandoned_events);
        for (const auto& event : events) {
            signal(event, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
    }

    cldnn::event::ptr make_ready_event() {
        std::lock_guard<std::mutex> lock(completion_mutex);
        OPENVINO_ASSERT(!completion_closed, "Lookahead completion registry is closed");
        auto event = std::make_shared<ReadyEvent>(context);
        ready_events.erase(std::remove_if(ready_events.begin(), ready_events.end(),
                                           [](const auto& weak) { return weak.expired(); }), ready_events.end());
        ready_events.push_back(event);
        return event;
    }

    void expire_locked() {
        bool changed = false;
        for (auto task = predictions.begin(); task != predictions.end();) {
            if (task->generation != generation || task->target_layer_idx <= cursor) {
                consume(*task);
                task = predictions.erase(task);
                changed = true;
            } else {
                ++task;
            }
        }
        for (auto job = speculative.begin(); job != speculative.end();) {
            if (job->generation != generation || job->layer <= cursor) {
                if (job->binding->cache->cancel_load(job->load)) {
                    abandoned.push_back(*job);
                } else {
                    demand.push_back(*job);
                }
                job = speculative.erase(job);
                changed = true;
            } else {
                ++job;
            }
        }
        if (changed) {
            wakeup->notify();
        }
    }

    void run() noexcept {
        try {
            for (;;) {
                std::optional<Job> selected;
                std::optional<PrefetchTask> prediction;
                std::shared_ptr<Binding> binding;
                {
                    std::unique_lock<std::mutex> lock(wakeup->mutex);
                    for (;;) {
                        uint64_t observed;
                        {
                            std::lock_guard<std::mutex> notification(wakeup->notification_mutex);
                            observed = wakeup->revision;
                        }
                        if (stop) {
                            discard_locked();
                            lock.unlock();
                            fail_abandoned();
                            return;
                        }
                        expire_locked();
                        if (!abandoned.empty()) {
                            active = true;
                            break;
                        }
                        std::vector<cldnn::event::ptr> dependencies;
                        for (const auto& job : demand) {
                            dependencies.push_back(job.load.last_use_event);
                        }
                        for (const auto& job : speculative) {
                            dependencies.push_back(job.load.last_use_event);
                        }
                        for (const auto& task : predictions) {
                            dependencies.push_back(task.prediction_ready_event);
                        }
                        lock.unlock();
                        std::unordered_map<const cldnn::event*, cl_int> statuses;
                        for (const auto& event : dependencies) {
                            statuses.emplace(event.get(), event_status(event));
                        }
                        lock.lock();
                        if (stop) {
                            continue;
                        }
                        expire_locked();
                        if (!abandoned.empty()) {
                            active = true;
                            break;
                        }
                        const auto terminal = [&](const cldnn::event::ptr& event) {
                            const auto found = statuses.find(event.get());
                            return found != statuses.end() && found->second <= CL_COMPLETE;
                        };
                        auto eligible = [&](const Job& job) { return terminal(job.load.last_use_event); };
                        auto urgent = std::find_if(demand.begin(), demand.end(), eligible);
                        if (urgent != demand.end()) {
                            selected = *urgent;
                            demand.erase(urgent);
                            active = true;
                            break;
                        }
                        auto hint = std::find_if(predictions.begin(), predictions.end(), [&](const PrefetchTask& task) {
                            return terminal(task.prediction_ready_event);
                        });
                        if (hint != predictions.end()) {
                            prediction = *hint;
                            predictions.erase(hint);
                            auto provider = prediction->target_provider.lock();
                            auto found = bindings.find(provider.get());
                            if (provider && found != bindings.end()) {
                                binding = found->second;
                            }
                            active = true;
                            break;
                        }
                        auto hint_load = std::find_if(speculative.begin(), speculative.end(), eligible);
                        if (hint_load != speculative.end() && demand.empty()) {
                            selected = *hint_load;
                            speculative.erase(hint_load);
                            active = true;
                            break;
                        }
                        std::unique_lock<std::mutex> notification(wakeup->notification_mutex);
                        if (observed != wakeup->revision) {
                            continue;
                        }
                        lock.unlock();
                        wakeup->notification_cv.wait(notification, [&] { return observed != wakeup->revision; });
                        notification.unlock();
                        lock.lock();
                    }
                }
                fail_abandoned();
                if (selected) {
                    try {
                        OPENVINO_ASSERT(event_status(selected->load.last_use_event) == CL_COMPLETE,
                                        "Lookahead consumer event failed");
                        if (selected->load.last_use_event) {
                            dynamic_cast<cldnn::ocl::ocl_stream&>(*dma).enqueue_external_barrier({selected->load.last_use_event});
                        }
                        auto& source = *selected->binding;
                        moe_otd::fill_weights_memory_async(*dma, source.config, source.offsets, *source.reader,
                                                           source.weights, {selected->load.expert_id}, {selected->load.slot},
                                                           nullptr, &staging);
                        finish(*selected, CL_COMPLETE);
                    } catch (...) {
                        finish(*selected, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
                        throw;
                    }
                } else if (prediction) {
                    std::vector<uint32_t> experts;
                    try {
                        OPENVINO_ASSERT(event_status(prediction->prediction_ready_event) == CL_COMPLETE,
                                        "Lookahead prediction event failed");
                        if (binding) {
                            const auto count = static_cast<size_t>(binding->config.top_k);
                            OPENVINO_ASSERT(prediction->prediction_event_mem && prediction->prediction_event_mem->count() >= count,
                                            "Invalid lookahead prediction buffer");
                            experts.resize(count);
                            prediction->prediction_event_mem->copy_to(*dma, experts.data(), 0, 0,
                                                                     count * sizeof(uint32_t), true);
                        }
                        consume(*prediction);
                    } catch (...) {
                        consume(*prediction);
                        throw;
                    }
                    for (auto expert : experts) {
                        OPENVINO_ASSERT(expert < static_cast<uint32_t>(binding->config.num_expert), "Invalid predicted expert ID");
                        std::optional<LRUCache::Load> load;
                        {
                            std::lock_guard<std::mutex> lock(wakeup->mutex);
                            if (stop || prediction->generation != generation || prediction->target_layer_idx <= cursor) {
                                break;
                            }
                            load = binding->cache->reserve_speculative(expert, binding->config.top_k,
                                [this] { return make_ready_event(); });
                        }
                        if (load) {
                            watch(load->last_use_event);
                            std::lock_guard<std::mutex> lock(wakeup->mutex);
                            speculative.push_back({binding, *load, {}, prediction->generation, prediction->target_layer_idx});
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(wakeup->mutex);
                    active = false;
                }
                wakeup->notify();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(wakeup->mutex);
                error = std::current_exception();
                stop = true;
                active = false;
                discard_locked();
            }
            fail_abandoned();
            wakeup->notify();
        }
    }

    std::shared_ptr<Wakeup> wakeup = std::make_shared<Wakeup>();
    cldnn::stream::ptr dma;
    cl::Context context;
    moe_otd::LookaheadStagingPool staging;
    std::unordered_map<OffloadExpertWeightProvider*, std::shared_ptr<Binding>> bindings;
    std::deque<Job> demand;
    std::deque<Job> speculative;
    std::vector<Job> abandoned;
    std::vector<std::weak_ptr<cldnn::event>> ready_events;
    std::mutex completion_mutex;
    bool completion_closed = false;
    std::vector<cldnn::event::ptr> abandoned_events;
    std::deque<PrefetchTask> predictions;
    uint64_t generation = 0;
    uint32_t cursor = 0;
    bool stop = false;
    bool active = false;
    std::exception_ptr error;
    std::thread worker;
};

ExpertPrefetcher::ExpertPrefetcher(cldnn::engine& engine, const cldnn::ExecutionConfig& config)
    : m_state(std::make_unique<State>(engine, config)) {}

ExpertPrefetcher::~ExpertPrefetcher() = default;

void ExpertPrefetcher::bind(const std::shared_ptr<OffloadExpertWeightProvider>& provider) {
    OPENVINO_ASSERT(provider && provider->_resident, "Lookahead provider must be bound to partial buffers");
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    if (!m_state->bindings.count(provider.get())) {
        auto binding = std::make_shared<State::Binding>();
        binding->cache = provider->_cache;
        binding->config = provider->_config;
        binding->offsets = provider->_weight_bin_offsets;
        binding->weights = *provider->_resident;
        binding->reader = std::make_unique<moe_otd::ParallelWeightReader>(provider->_weights_path);
        m_state->bindings.emplace(provider.get(), std::move(binding));
    }
}

cldnn::event::ptr ExpertPrefetcher::make_ready_event() {
    return m_state->make_ready_event();
}

void ExpertPrefetcher::submit_prefetch(PrefetchTask task) {
    rethrow_if_failed();
    m_state->watch(task.prediction_ready_event);
    {
        std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
        OPENVINO_ASSERT(!m_state->stop, "Lookahead prefetcher is stopped");
        m_state->predictions.push_back(std::move(task));
    }
    m_state->wakeup->notify();
}

cldnn::event::ptr ExpertPrefetcher::demand_async(OffloadExpertWeightProvider& provider,
                                              const std::vector<uint32_t>& missing_ids) {
    rethrow_if_failed();
    if (missing_ids.empty()) {
        return {};
    }
    auto batch = std::make_shared<State::Batch>(State::Batch{missing_ids.size(), make_ready_event()});
    std::vector<LRUCache::Load> loads;
    for (auto expert : missing_ids) {
        auto load = provider.cache().get_pending_load(expert);
        OPENVINO_ASSERT(load, "Demand expert was not reserved atomically");
        m_state->watch(load->last_use_event);
        loads.push_back(*load);
    }
    {
        std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
        OPENVINO_ASSERT(!m_state->stop, "Lookahead prefetcher is stopped");
        auto binding = m_state->bindings.at(&provider);
        for (const auto& load : loads) {
            m_state->demand.push_back({binding, load, batch});
        }
    }
    m_state->wakeup->notify();
    return batch->ready;
}

void ExpertPrefetcher::promote(OffloadExpertWeightProvider& provider, const std::vector<uint32_t>& topk_ids) {
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    const auto binding = m_state->bindings.at(&provider);
    for (auto job = m_state->speculative.begin(); job != m_state->speculative.end();) {
        if (job->binding == binding && std::find(topk_ids.begin(), topk_ids.end(), job->load.expert_id) != topk_ids.end()) {
            m_state->demand.push_back(*job);
            job = m_state->speculative.erase(job);
        } else {
            ++job;
        }
    }
    m_state->wakeup->notify();
}

void ExpertPrefetcher::advance(uint64_t generation, uint32_t layer_idx) {
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    m_state->generation = generation;
    m_state->cursor = layer_idx;
    m_state->wakeup->notify();
}

void ExpertPrefetcher::rethrow_if_failed() const {
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    if (m_state->error) {
        std::rethrow_exception(m_state->error);
    }
}

void ExpertPrefetcher::cancel_all() {
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    m_state->stop = true;
    m_state->wakeup->notify();
}

void ExpertPrefetcher::drain() {
    std::unique_lock<std::mutex> lock(m_state->wakeup->mutex);
    m_state->wakeup->cv.wait(lock, [&] {
        return m_state->stop || (!m_state->active && m_state->demand.empty() &&
                                m_state->speculative.empty() && m_state->predictions.empty());
    });
    if (m_state->error) {
        std::rethrow_exception(m_state->error);
    }
    OPENVINO_ASSERT(!m_state->stop, "Lookahead prefetcher is stopped");
}

}