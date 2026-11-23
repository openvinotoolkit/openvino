// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

namespace {
// Background reads of prefetched experts from the weights file
class ReadPool {
public:
    static ReadPool& get() {
        static ReadPool pool(8);
        return pool;
    }
    std::future<void> submit(std::function<void()> task) {
        auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
        auto future = packaged->get_future();
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _tasks.push([packaged] {
                (*packaged)();
            });
        }
        _cv.notify_one();
        return future;
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& t : _threads) {
            t.join();
        }
    }

private:
    explicit ReadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; i++) {
            _threads.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(_mutex);
                        _cv.wait(lock, [this] {
                            return _stop || !_tasks.empty();
                        });
                        if (_stop && _tasks.empty()) {
                            return;
                        }
                        task = std::move(_tasks.front());
                        _tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    std::vector<std::thread> _threads;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _stop = false;
};
}  // namespace

bool is_per_expert(const cldnn::layout& layout, size_t num_expert) {
    return layout.is_static() && layout.count() > 1 && !layout.get_shape().empty() && layout.get_shape()[0] == num_expert;
}

bool has_contiguous_expert_slices(const cldnn::layout& layout, size_t num_expert) {
    return cldnn::format::is_simple_data_format(layout.format) && layout.format.order().front() == 'b' && layout.bytes_count() % num_expert == 0;
}

cldnn::layout get_slot_layout(const cldnn::layout& input, size_t capacity, size_t num_expert) {
    OPENVINO_ASSERT(has_contiguous_expert_slices(input, num_expert), "OTD: expert slices of ", input.to_short_string(), " are not contiguous");
    auto slots_shape = input.get_shape();
    slots_shape[0] = capacity;
    return cldnn::layout(slots_shape, input.data_type, input.format);
}

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                                                      const RoutedWeightViews& /*inputs*/,
                                                                                      const RoutedWeightViews& /*slots*/,
                                                                                      cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert,
                                                 const RoutedWeightViews& /*inputs*/,
                                                 const RoutedWeightViews& /*slots*/,
                                                 cldnn::stream& /*stream*/) {
    return expert;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::filesystem::path weights_path)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _cache(std::make_shared<LRUCache>(capacity)),
      _weight_reader(_weights_path) {}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& /*resident*/) {
    _cache->set_initialized();
    _bound = true;
    GPU_DEBUG_TRACE << "MOE OTD bind: resident_slots=" << _capacity << " total_experts=" << _config.num_expert
                    << " shared_expert=" << (_config.num_shared_expert > 0 ? 1 : 0) << std::endl;
}

void OffloadExpertWeightProvider::reset_cache_if_slots_changed(const RoutedWeightViews& slots) {
    std::vector<const void*> handles;
    for (size_t i = 0; i < 3; i++) {
        for (const auto& mem : {slots.weight[i], slots.scale[i], slots.zp[i]}) {
            if (mem) {
                handles.push_back(mem->get_internal_params().mem);
            }
        }
    }
    if (handles != _slot_handles) {
        _slot_handles = std::move(handles);
        _cache = std::make_shared<LRUCache>(_capacity);
        _cache->set_initialized();
    }
}

void OffloadExpertWeightProvider::load(uint32_t expert, size_t slot, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) {
    OPENVINO_ASSERT(_weight_bin_offsets.size() == cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, "Unexpected number of MOE weight offsets");
    const size_t num_expert = _config.num_expert;
    // The weights are streamed as stored in the weights file, the scales and zero points are copied as converted by the pipeline
    std::vector<moe_otd::SlicedTensor> tensors;
    for (size_t i = 0; i < 3; i++) {
        if (slots.weight[i]) {
            tensors.push_back({slots.weight[i], slots.weight[i]->get_layout().bytes_count() / num_expert, nullptr, _weight_bin_offsets[i]});
        }
        for (const auto& [slot_mem, source] : {std::pair{slots.scale[i], inputs.scale[i]}, std::pair{slots.zp[i], inputs.zp[i]}}) {
            if (slot_mem) {
                tensors.push_back({slot_mem, source->get_layout().bytes_count() / num_expert, source});
            }
        }
    }
    moe_otd::fill_weights_memory(stream, _weight_reader, tensors, expert, slot);
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts,
                                                                                     const RoutedWeightViews& inputs,
                                                                                     const RoutedWeightViews& slot_buffers,
                                                                                     cldnn::stream& stream) {
    reset_cache_if_slots_changed(slot_buffers);

    // Deduplicate to check capacity
    std::unordered_map<uint32_t, size_t> expert_to_slot;
    expert_to_slot.reserve(experts.size());

    std::vector<size_t> slots(experts.size());
    auto* perf = moe_otd::get_perf_counters();

    for (size_t i = 0; i < experts.size(); i++) {
        const uint32_t expert = experts[i];
        auto it = expert_to_slot.find(expert);
        if (it != expert_to_slot.end()) {
            slots[i] = it->second;
            continue;
        }

        // Check if we would exceed capacity
        if (expert_to_slot.size() >= _capacity) {
            return std::nullopt;
        }

        const auto item = _cache->get_lru_item(expert);
        const auto slot = item.first;

        if (item.second) {
            if (perf) {
                perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            if (perf) {
                perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
            }
            if (!consume_pending(expert, slot, inputs, slot_buffers, stream)) {
                load(expert, slot, inputs, slot_buffers, stream);
            }
            _cache->set_filled(slot);
        }

        expert_to_slot.emplace(expert, slot);
        slots[i] = slot;
    }

    return slots;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, const RoutedWeightViews& inputs, const RoutedWeightViews& slots, cldnn::stream& stream) {
    reset_cache_if_slots_changed(slots);
    auto* perf = moe_otd::get_perf_counters();

    const auto item = _cache->get_lru_item(expert);
    const auto slot = item.first;

    if (item.second) {
        if (perf) {
            perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (perf) {
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        if (!consume_pending(expert, slot, inputs, slots, stream)) {
            load(expert, slot, inputs, slots, stream);
        }
        _cache->set_filled(slot);
    }

    return slot;
}

bool OffloadExpertWeightProvider::consume_pending(uint32_t expert,
                                                  size_t slot,
                                                  const RoutedWeightViews& inputs,
                                                  const RoutedWeightViews& slots,
                                                  cldnn::stream& stream) {
    auto it = _pending.find(expert);
    if (it == _pending.end()) {
        return false;
    }
    auto pending = it->second;
    _pending.erase(it);
    for (auto& read : pending->reads) {
        read.wait();
    }
    const bool blocking = stream.get_queue_type() != cldnn::QueueTypes::in_order;
    const size_t num_expert = _config.num_expert;
    for (size_t i = 0; i < 3; i++) {
        if (slots.weight[i] && !pending->payloads[i].empty()) {
            const size_t size = pending->payloads[i].size();
            slots.weight[i]->copy_from(stream, pending->payloads[i].data(), 0, slot * size, size, blocking);
        }
        for (const auto& [slot_mem, source] : {std::pair{slots.scale[i], inputs.scale[i]}, std::pair{slots.zp[i], inputs.zp[i]}}) {
            if (slot_mem) {
                const size_t size = source->get_layout().bytes_count() / num_expert;
                slot_mem->copy_from(stream, *source, expert * size, slot * size, size, blocking);
            }
        }
    }
    _inflight.push_back(std::move(pending));
    if (auto* perf = moe_otd::get_perf_counters()) {
        perf->les_prefetch_used.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void OffloadExpertWeightProvider::prefetch(const std::vector<uint32_t>& experts,
                                           const RoutedWeightViews& /*inputs*/,
                                           const RoutedWeightViews& slots,
                                           cldnn::stream& /*stream*/) {
    if (!_bound || !slots.weight[0] || !slots.weight[1] || !slots.weight[2]) {
        return;  // the slot buffers are allocated at the first execution of the layer
    }
    auto* perf = moe_otd::get_perf_counters();
    // Predictions of the previous token that were not used: their reads are done long ago
    for (auto& [expert, pending] : _pending) {
        for (auto& read : pending->reads) {
            read.wait();
        }
    }
    _pending.clear();
    // The device copies of the consumed prefetches are complete: every MoE layer synchronizes the stream
    _inflight.clear();

    // Only the file reads start here, on background threads: the layer copies an expert to a slot when the router
    // selects it, so a wrong prediction costs neither a slot nor a device copy
    const size_t num_expert = _config.num_expert;
    const size_t file_size = _weight_reader.file_size();  // opens the file before the background reads
    const size_t max_loads = _capacity / 2;
    for (uint32_t expert : experts) {
        if (expert >= num_expert || _pending.size() >= max_loads || _pending.count(expert)) {
            continue;
        }
        if (_cache->contains_filled(expert)) {
            if (perf) {
                perf->les_prefetch_resident.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        auto pending = std::make_shared<PendingExpert>();
        pending->payloads.resize(3);
        for (size_t i = 0; i < 3; i++) {
            const size_t size = slots.weight[i]->get_layout().bytes_count() / num_expert;
            const size_t offset = _weight_bin_offsets[i] + expert * size;
            OPENVINO_ASSERT(offset + size <= file_size, "OTD prefetch: read range out of the weights file");
            pending->payloads[i].resize(size);
            char* dst = reinterpret_cast<char*>(pending->payloads[i].data());
            pending->reads.push_back(ReadPool::get().submit([this, dst, size, offset] {
                _weight_reader.read(dst, size, offset);
            }));
        }
        _pending.emplace(expert, std::move(pending));
        if (perf) {
            perf->les_prefetch_loads.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

}  // namespace ov::intel_gpu::ocl::moe
