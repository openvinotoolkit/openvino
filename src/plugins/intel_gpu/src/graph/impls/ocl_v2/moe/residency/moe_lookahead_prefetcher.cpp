#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_prefetcher.hpp"

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <string_view>
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

size_t worker_count() {
    const auto* value = std::getenv("MOE_LES_WORKERS");
    if (!value || std::string_view(value) == "1") {
        return 1;
    }
    if (std::string_view(value) == "4") {
        return 4;
    }
    OPENVINO_ASSERT(std::string_view(value) == "8", "MOE_LES_WORKERS must be 1, 4 or 8");
    return 8;
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
        std::vector<moe_otd::SlicedTensor> tensors;
        std::unique_ptr<moe_otd::ParallelWeightReader> reader;
    };
    struct Batch {
        Batch(size_t count, cldnn::event::ptr event) : remaining(count), ready(std::move(event)) {}

        std::atomic<size_t> remaining;
        cldnn::event::ptr ready;
    };
    struct Job {
        std::shared_ptr<Binding> binding;
        LRUCache::Load load;
        std::shared_ptr<Batch> batch;
        uint64_t generation = 0;
        uint32_t layer = 0;
    };
    struct Worker {
        cldnn::stream::ptr dma;
        moe_otd::LookaheadStagingPool staging;
        std::thread thread;
    };

    State(cldnn::engine& engine, const cldnn::ExecutionConfig& config, cldnn::stream::ptr compute_stream)
        : context(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context()) {
        const auto* use_new_stream = std::getenv("MOE_LES_USE_NEW_STREAM");
        workers.resize(worker_count());
        shares_compute_queue = compute_stream &&
                               !(use_new_stream && std::string_view(use_new_stream) == "1");
        for (auto& worker : workers) {
            worker.dma = shares_compute_queue ? compute_stream : moe_otd::create_copy_stream(engine, config);
            for (auto& chunk : worker.staging.chunks) {
                chunk.memory = engine.allocate_memory(
                    cldnn::layout{cldnn::data_types::u8, cldnn::format::bfyx,
                                  {1, 1, 1, static_cast<int32_t>(moe_otd::LookaheadStagingPool::chunk_bytes)}},
                    cldnn::allocation_type::usm_host, false);
            }
        }
        try {
            for (auto& worker : workers) {
                worker.thread = std::thread([this, &worker] { run(worker); });
            }
        } catch (...) {
            stop_workers();
            throw;
        }
    }

    ~State() {
        stop_workers();
        if (!shares_compute_queue) {
            for (auto& worker : workers) {
                try {
                    worker.dma->finish();
                } catch (const std::exception& exception) {
                    GPU_DEBUG_INFO << "Lookahead DMA teardown failed: " << exception.what() << std::endl;
                }
            }
        }
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(wakeup->mutex);
            stop = true;
        }
        wakeup->notify();
        for (auto& worker : workers) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
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
        } else {
            job.binding->cache->fail_load(job.load);
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
        std::vector<cldnn::event::ptr> events;
        {
            std::lock_guard<std::mutex> lock(wakeup->mutex);
            jobs.swap(abandoned);
            events.swap(abandoned_events);
        }
        for (auto& job : jobs) {
            finish(job, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
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

    void run(Worker& worker) noexcept {
        bool running_job = false;
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
                            if (active == 0) {
                                discard_locked();
                            }
                            lock.unlock();
                            fail_abandoned();
                            return;
                        }
                        expire_locked();
                        if (!abandoned.empty()) {
                            ++active;
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
                            ++active;
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
                            ++active;
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
                            ++active;
                            break;
                        }
                        auto hint_load = std::find_if(speculative.begin(), speculative.end(), eligible);
                        if (hint_load != speculative.end() && demand.empty()) {
                            selected = *hint_load;
                            speculative.erase(hint_load);
                            ++active;
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
                running_job = true;
                fail_abandoned();
                if (selected) {
                    try {
                        OPENVINO_ASSERT(event_status(selected->load.last_use_event) == CL_COMPLETE,
                                        "Lookahead consumer event failed");
                        auto& source = *selected->binding;
                        OPENVINO_ASSERT(source.cache->start_load(selected->load), "Lookahead slot reservation is stale");
                        moe_otd::fill_weights_memory_async(*worker.dma, *source.reader, source.tensors,
                                                           {selected->load.expert_id}, {selected->load.slot},
                                                           nullptr, &worker.staging);
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
                            if (prediction->cpu_indices) {
                                OPENVINO_ASSERT(!prediction->cpu_indices->empty(), "Empty prediction CPU buffer");
                                experts = *prediction->cpu_indices;
                            } else if (prediction->prediction_event_mem->get_allocation_type() == cldnn::allocation_type::usm_host) {
                                cldnn::mem_lock<uint32_t, cldnn::mem_lock_type::read> indices(prediction->prediction_event_mem,
                                                                                              *worker.dma);
                                std::copy(indices.data(), indices.data() + count, experts.begin());
                            } else {
                                prediction->prediction_event_mem->copy_to(*worker.dma, experts.data(), 0, 0,
                                                                         count * sizeof(uint32_t), true);
                            }
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
                                [this] { return make_ready_event(); }, experts);
                            if (load) {
                                speculative.push_back({binding, *load, {}, prediction->generation, prediction->target_layer_idx});
                            }
                        }
                        if (load) {
                            watch(load->last_use_event);
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(wakeup->mutex);
                    --active;
                    running_job = false;
                }
                wakeup->notify();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(wakeup->mutex);
                if (!error) {
                    error = std::current_exception();
                }
                stop = true;
                if (running_job) {
                    --active;
                }
                if (active == 0) {
                    discard_locked();
                }
            }
            fail_abandoned();
            wakeup->notify();
        }
    }

    std::shared_ptr<Wakeup> wakeup = std::make_shared<Wakeup>();
    bool shares_compute_queue = false;
    cl::Context context;
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
    size_t active = 0;
    std::exception_ptr error;
    std::vector<Worker> workers;
};

ExpertPrefetcher::ExpertPrefetcher(cldnn::engine& engine, const cldnn::ExecutionConfig& config, cldnn::stream::ptr compute_stream)
    : m_state(std::make_unique<State>(engine, config, std::move(compute_stream))) {}

ExpertPrefetcher::~ExpertPrefetcher() = default;

bool ExpertPrefetcher::wait_for_uploads_on_shared_queue(const std::vector<cldnn::event::ptr>& events) const {
    if (m_state->shares_compute_queue && !events.empty()) {
        m_state->workers.front().dma->flush();
        for (const auto& event : events) {
            event->wait();
        }
    }
    return m_state->shares_compute_queue;
}

void ExpertPrefetcher::bind(const std::shared_ptr<OffloadExpertWeightProvider>& provider) {
    OPENVINO_ASSERT(provider && provider->_resident, "Lookahead provider must be bound to partial buffers");
    std::lock_guard<std::mutex> lock(m_state->wakeup->mutex);
    if (!m_state->bindings.count(provider.get())) {
        auto binding = std::make_shared<State::Binding>();
        binding->cache = provider->_cache;
        binding->config = provider->_config;
        binding->tensors = provider->_load_tensors;
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
    auto batch = std::make_shared<State::Batch>(missing_ids.size(), make_ready_event());
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
        return m_state->stop || (m_state->active == 0 && m_state->abandoned.empty() && m_state->demand.empty() &&
                                m_state->speculative.empty() && m_state->predictions.empty());
    });
    if (m_state->error) {
        std::rethrow_exception(m_state->error);
    }
    OPENVINO_ASSERT(!m_state->stop, "Lookahead prefetcher is stopped");
}

}