// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <unordered_map>
#include <utility>
#include <chrono>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"
#include "intel_gpu/runtime/les_context.hpp"
#include "runtime/ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl::moe {

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& /*stream*/) {
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

OffloadExpertWeightProvider::~OffloadExpertWeightProvider() {
    if (_prefetch_pending) {
        resolve_prefetch({}, false);
    }
    if (_upload_completion.valid()) {
        _upload_completion.wait();
    }
}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& resident) {
    _resident = &resident;
    _cache->set_initialized();
    _bound = true;
    GPU_DEBUG_TRACE << "MOE OTD bind: resident_slots=" << _capacity << " total_experts=" << _config.num_expert
                    << " shared_expert=" << (_config.num_shared_expert > 0 ? 1 : 0) << std::endl;
}

std::shared_ptr<IExpertWeightProvider> OffloadExpertWeightProvider::clone() const {
    return std::make_shared<OffloadExpertWeightProvider>(_capacity, _config, _weight_bin_offsets, _weights_path);
}

void OffloadExpertWeightProvider::set_les_context(cldnn::les_context* context, uint64_t epoch) {
    if (_prefetch_pending && (_les_context != context || _les_epoch != epoch)) {
        resolve_prefetch({}, _les_context == context);
    }
    if (_les_context != context) {
        release();
        _reservations = context ? std::make_unique<ExpertSlotReservations>(_capacity) : nullptr;
        _demand_shadow = context ? std::make_unique<ExpertSlotReservations>(_capacity) : nullptr;
        _les_context = context;
    }
    _les_epoch = epoch;
}

void OffloadExpertWeightProvider::set_consumer(cldnn::event::ptr consumer) {
    if (_les_context) {
        _consumer = std::move(consumer);
    }
}

void OffloadExpertWeightProvider::release() {
    if (_prefetch_pending) {
        resolve_prefetch({});
    }
    std::exception_ptr error;
    if (_upload_completion.valid()) {
        try {
            _upload_completion.get();
        } catch (...) {
            error = std::current_exception();
        }
        _upload_completion = {};
    }
    _upload_events.reset();
    if (_consumer) {
        try {
            _consumer->wait();
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
        _consumer.reset();
    }
    if (_reservation) {
        if (!_speculative_reservation || !_reservations->rollback(*_reservation)) {
            _reservations->release(*_reservation);
        }
        _reservation.reset();
    }
    _speculative_reservation = false;
    if (_shadow_reservation) {
        _demand_shadow->release(*_shadow_reservation);
        _shadow_reservation.reset();
    }
    _demand_finished = false;
    _prefetch_error = nullptr;
    if (error) {
        std::rethrow_exception(error);
    }
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::acquire_les(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    auto demand = begin_les_demand(experts);
    if (!demand) {
        return std::nullopt;
    }
    finish_les_demand(stream);
    return std::move(demand->slots);
}

std::optional<ExpertDemand> OffloadExpertWeightProvider::begin_les_demand(const std::vector<uint32_t>& experts) {
    OPENVINO_ASSERT(_les_context && _reservations, "LES demand requires a context");
    if (_prefetch_pending) {
        resolve_prefetch(experts);
    }
    release();
    _reservation = _reservations->reserve(experts, _les_epoch);
    if (!_reservation) {
        return std::nullopt;
    }
    try {
        _shadow_reservation = _demand_shadow->reserve(experts, _les_epoch);
        OPENVINO_ASSERT(_shadow_reservation, "LES demand shadow reservation failed");
        OPENVINO_ASSERT(_resident && _les_context->get_transfer_stream(), "LES provider is not bound to a transfer lane");
        ExpertDemand demand{_reservation->slots, _reservation->ready_pairs, _reservation->missing_pairs};
        _upload_events = std::make_shared<std::vector<cldnn::event::ptr>>();
        if (!_reservation->missing_experts.empty()) {
            auto task = _les_context->submit(_les_epoch,
            [this, events = _upload_events, transfer = _les_context->get_transfer_stream(), weights = *_resident,
                 missing = _reservation->missing_experts, slots = _reservation->missing_slots]() mutable {
                    if (auto* perf = moe_otd::get_perf_counters()) {
                        perf->les_worker_tasks.fetch_add(1, std::memory_order_relaxed);
                    }
                    *events = moe_otd::fill_weights_memory_async(*transfer, _config, _weight_bin_offsets, _weight_reader,
                                                               weights, missing, slots);
                });
            _upload_completion = std::move(task.completion);
        }
        return demand;
    } catch (...) {
        release();
        throw;
    }
}

void OffloadExpertWeightProvider::finish_les_demand(cldnn::stream& stream) {
    OPENVINO_ASSERT(_reservation && !_demand_finished, "LES demand is missing or already finished");
    try {
        if (_upload_completion.valid()) {
            _upload_completion.get();
        }
        if (_prefetch_error) {
            auto error = _prefetch_error;
            release();
            std::rethrow_exception(error);
        }
        dynamic_cast<cldnn::ocl::ocl_stream&>(stream).enqueue_external_barrier(*_upload_events);
        OPENVINO_ASSERT(_reservations->complete(*_reservation), "LES reservation became stale before GPUReady");
        OPENVINO_ASSERT(_demand_shadow->complete(*_shadow_reservation), "LES shadow reservation became stale");
        if (auto* perf = moe_otd::get_perf_counters()) {
            auto unique_slots = _reservation->slots;
            std::sort(unique_slots.begin(), unique_slots.end());
            unique_slots.erase(std::unique(unique_slots.begin(), unique_slots.end()), unique_slots.end());
            perf->gpu_hits.fetch_add(unique_slots.size() - _reservation->missing_slots.size(), std::memory_order_relaxed);
            perf->gpu_misses.fetch_add(_reservation->missing_slots.size(), std::memory_order_relaxed);
            for (auto expert : _reservation->missing_experts) {
                if (std::find(_shadow_reservation->missing_experts.begin(), _shadow_reservation->missing_experts.end(), expert) ==
                    _shadow_reservation->missing_experts.end()) {
                    perf->les_prefetch_extra_misses.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        _demand_finished = true;
        _upload_completion = {};
        _upload_events.reset();
    } catch (...) {
        release();
        throw;
    }
}

void OffloadExpertWeightProvider::resolve_prefetch(const std::vector<uint32_t>& experts, bool schedule) {
    auto* perf = moe_otd::get_perf_counters();
    const auto& missing = _reservation->missing_experts;
    const auto used = std::count_if(missing.begin(), missing.end(), [&](uint32_t expert) {
        return std::find(experts.begin(), experts.end(), expert) != experts.end();
    });
    const bool late = _upload_completion.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    if (schedule) {
        if (used) {
            if (_les_context->promote(*_prefetch_task) && perf) {
                perf->les_prefetch_promoted.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            _les_context->cancel_pending(*_prefetch_task);
        }
    }
    bool completed = false;
    try {
        _upload_completion.get();
        if (_prefetch_error) {
            std::rethrow_exception(_prefetch_error);
        }
        completed = _reservations->complete(*_reservation);
    } catch (const cldnn::les_task_cancelled&) {
    } catch (...) {
        if (perf) {
            perf->les_prefetch_failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const auto useful = completed ? used * _prefetch_expert_bytes : 0;
    if (perf) {
        perf->les_prefetch_useful_bytes.fetch_add(useful, std::memory_order_relaxed);
        perf->les_prefetch_useless_bytes.fetch_add(_prefetch_uploaded_bytes - useful, std::memory_order_relaxed);
        perf->les_prefetch_late_bytes.fetch_add(late ? useful : 0, std::memory_order_relaxed);
    }
    if (_prefetch_uploaded_bytes != 0 || !_reservations->rollback(*_reservation)) {
        _reservations->discard_unused(*_reservation, completed ? experts : std::vector<uint32_t>{});
        _reservations->release(*_reservation);
    }
    _reservation.reset();
    _speculative_reservation = false;
    _upload_completion = {};
    _upload_events.reset();
    _prefetch_error = nullptr;
    _prefetch_task.reset();
    _prefetch_pending = false;
}

void OffloadExpertWeightProvider::prefetch_les_demand(const std::vector<uint32_t>& experts, cldnn::stream&) {
    if (!_les_context || !_reservations) {
        return;
    }
    try {
        auto* perf = moe_otd::get_perf_counters();
        if (_prefetch_pending) {
            if (perf) {
                const bool duplicate = std::all_of(experts.begin(), experts.end(), [&](uint32_t expert) {
                    return std::find(_reservation->missing_experts.begin(), _reservation->missing_experts.end(), expert) !=
                           _reservation->missing_experts.end();
                });
                (duplicate ? perf->les_prefetch_deduplicated : perf->les_prefetch_skipped)
                    .fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        release();
        _prefetch_error = nullptr;
        OPENVINO_ASSERT(_resident && _les_context->get_transfer_stream(), "LES provider is not bound to a transfer lane");
        _prefetch_expert_bytes = 0;
        for (const auto& tensor : {_resident->gate_w, _resident->up_w, _resident->down_w,
                                   _resident->gate_s, _resident->up_s, _resident->down_s,
                                   _resident->gate_z, _resident->up_z, _resident->down_z}) {
            if (tensor) {
                _prefetch_expert_bytes += tensor->get_layout().bytes_count() / _config.num_expert;
            }
        }
        std::vector<uint32_t> selected;
        const auto budget = moe_otd::LesSpeculationPolicy::expert_budget(_capacity, _prefetch_expert_bytes);
        for (auto expert : experts) {
            if (selected.size() == budget) {
                break;
            }
            if (std::find(selected.begin(), selected.end(), expert) == selected.end()) {
                selected.push_back(expert);
            }
        }
        if (selected.empty()) {
            if (perf) {
                perf->les_prefetch_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        _reservation = _reservations->reserve(selected, _les_epoch, true);
        _speculative_reservation = _reservation.has_value();
        if (!_reservation) {
            if (perf) {
                perf->les_prefetch_skipped.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
        const auto prefetched_slots = _reservation->missing_experts.size();
        if (!prefetched_slots) {
            release();
            return;
        }
        _prefetch_uploaded_bytes = 0;
        _upload_events = std::make_shared<std::vector<cldnn::event::ptr>>();
        if (!_reservation->missing_experts.empty()) {
            auto task = _les_context->try_submit_speculative(_les_epoch,
                [this, events = _upload_events, transfer = _les_context->get_transfer_stream(), weights = *_resident,
                 missing = _reservation->missing_experts, slots = _reservation->missing_slots]() mutable {
                    try {
                        *events = moe_otd::fill_weights_memory_async(*transfer, _config, _weight_bin_offsets, _weight_reader,
                                                                     weights, missing, slots, &_prefetch_uploaded_bytes);
                    } catch (...) {
                        _prefetch_error = std::current_exception();
                    }
                });
            if (!task) {
                release();
                if (perf) {
                    perf->les_prefetch_skipped.fetch_add(1, std::memory_order_relaxed);
                }
                return;
            }
            _prefetch_task = std::move(task);
            _upload_completion = _prefetch_task->completion;
            _prefetch_pending = true;
        }
        if (perf) {
            perf->les_prefetch_experts.fetch_add(prefetched_slots, std::memory_order_relaxed);
        }
    } catch (const cldnn::les_task_cancelled&) {
        try {
            release();
        } catch (...) {
        }
    } catch (const std::exception& error) {
        try {
            release();
        } catch (...) {
        }
        if (auto* perf = moe_otd::get_perf_counters()) {
            perf->les_prefetch_failures.fetch_add(1, std::memory_order_relaxed);
        }
        GPU_DEBUG_INFO << "[LES_PREDICT_PREFETCH] speculative demand skipped: " << error.what() << std::endl;
    } catch (...) {
        try {
            release();
        } catch (...) {
        }
        if (auto* perf = moe_otd::get_perf_counters()) {
            perf->les_prefetch_failures.fetch_add(1, std::memory_order_relaxed);
        }
        GPU_DEBUG_INFO << "[LES_PREDICT_PREFETCH] speculative demand skipped: unknown failure" << std::endl;
    }
}

void OffloadExpertWeightProvider::fill_routed_weight_views(cldnn::moe_weights& /*weights*/, RoutedWeightViews& views) {
    // Offloaded: point at the LRU resident buffers (not the original weight inputs).
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    views.weight[0] = _resident->gate_w;
    views.scale[0] = _resident->gate_s;
    views.zp[0] = _resident->gate_z;
    views.weight[1] = _resident->up_w;
    views.scale[1] = _resident->up_s;
    views.zp[1] = _resident->up_z;
    views.weight[2] = _resident->down_w;
    views.scale[2] = _resident->down_s;
    views.zp[2] = _resident->down_z;
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    if (_les_context) {
        return acquire_les(experts, stream);
    }
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
            OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
            moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, {expert}, {slot});
            _cache->set_filled(slot);
        }

        expert_to_slot.emplace(expert, slot);
        slots[i] = slot;
    }

    return slots;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& stream) {
    if (_les_context) {
        auto lease = acquire_les({expert}, stream);
        OPENVINO_ASSERT(lease, "LES single-expert reservation is blocked");
        return lease->front();
    }
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
        OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
        moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, {expert}, {slot});
        _cache->set_filled(slot);
    }

    return slot;
}

}  // namespace ov::intel_gpu::ocl::moe
