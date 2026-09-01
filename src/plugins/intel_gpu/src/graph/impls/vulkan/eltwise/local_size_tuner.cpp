// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "local_size_tuner.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"

namespace cldnn::vulkan::eltwise_detail {
namespace {

constexpr size_t samples_per_candidate = 3;
constexpr size_t stabilization_inferences = 20;

uint64_t median_sample(std::vector<uint64_t> samples) {
    const auto middle = samples.begin() + samples.size() / 2;
    std::nth_element(samples.begin(), middle, samples.end());
    return *middle;
}

bool diagnostics_enabled() {
    const auto* value = std::getenv("OV_GPU_VULKAN_LOCAL_SIZE_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

}  // namespace

uint32_t select_portable_local_work_group_size(uint32_t invocation_count, uint64_t device_max_work_group_size) {
    const auto limit = static_cast<uint32_t>(std::min<uint64_t>(portable_local_work_group_size_limit, device_max_work_group_size));
    OPENVINO_ASSERT(limit > 0, "[GPU][Vulkan] Device reports a zero maximum work-group size");

    uint32_t local_size = 1;
    while (local_size < invocation_count && local_size <= limit / 2) {
        local_size *= 2;
    }
    return local_size;
}

LocalSizeTuner::Selection::Selection(LocalSizeTuner& owner, Action selected_action, std::unique_lock<std::mutex> lock)
    : _owner(&owner),
      _action(selected_action),
      _lock(std::move(lock)) {}

uint32_t LocalSizeTuner::Selection::local_size() const {
    return _action.local_size;
}

bool LocalSizeTuner::Selection::requires_prewarm() const {
    return _action.prewarm;
}

bool LocalSizeTuner::Selection::requires_measurement() const {
    return _action.measure;
}

const std::vector<uint32_t>& LocalSizeTuner::Selection::candidates() const {
    OPENVINO_ASSERT(_owner != nullptr && _action.prewarm && _lock.owns_lock(),
                    "[GPU][Vulkan] Eltwise local-size candidates require an active prewarm selection");
    return _owner->_candidates;
}

uint32_t LocalSizeTuner::Selection::complete_prewarm() {
    OPENVINO_ASSERT(_owner != nullptr && _action.prewarm && _lock.owns_lock(), "[GPU][Vulkan] Eltwise local-size prewarm completion has no active selection");
    _owner->_prewarmed = true;
    const auto fallback = _owner->_fallback;
    _lock.unlock();
    return fallback;
}

void LocalSizeTuner::Selection::complete_measurement(uint64_t elapsed_nanoseconds) {
    OPENVINO_ASSERT(_owner != nullptr && _action.measure && _lock.owns_lock(),
                    "[GPU][Vulkan] Eltwise local-size measurement completion has no active selection");
    _owner->_samples[_action.candidate_index].push_back(elapsed_nanoseconds);
    _lock.unlock();
}

LocalSizeTuner::Selection LocalSizeTuner::select_uncached(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility) {
    std::unique_lock lock(_mutex);
    auto selected_action = next_action(invocation_count, info, require_exact_divisibility);
    if (!selected_action.prewarm && !selected_action.measure) {
        lock.unlock();
    }
    return Selection(*this, selected_action, std::move(lock));
}

void LocalSizeTuner::initialize(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility) {
    _invocation_count = invocation_count;
    _fallback = select_portable_local_work_group_size(invocation_count, info.max_work_group_size);
    _selected = _fallback;
    _candidates = {_fallback};

    if (!info.supported_simd_sizes.empty() && info.supported_simd_sizes.front() != 0) {
        const auto limit = static_cast<uint32_t>(std::min<uint64_t>(info.max_work_group_size, std::numeric_limits<uint32_t>::max()));
        const auto subgroup_size = info.supported_simd_sizes.front();
        const auto add_candidate = [&](uint32_t candidate) {
            if (candidate >= subgroup_size && candidate % subgroup_size == 0 && candidate <= limit && candidate <= invocation_count &&
                (!require_exact_divisibility || invocation_count % candidate == 0)) {
                _candidates.push_back(candidate);
            }
        };
        if (subgroup_size <= _fallback / 2) {
            add_candidate((_fallback / 2 / subgroup_size) * subgroup_size);
        }
        if (_fallback <= limit / 2) {
            add_candidate(((_fallback * 2 + subgroup_size - 1) / subgroup_size) * subgroup_size);
        }
        std::sort(_candidates.begin(), _candidates.end());
        _candidates.erase(std::unique(_candidates.begin(), _candidates.end()), _candidates.end());
    }

    _samples.assign(_candidates.size(), {});
    _fallback_observations = 0;
    _prewarmed = false;
    _decided = _candidates.size() == 1;
    _cached_selected.store(_decided ? _fallback : 0, std::memory_order_release);
}

void LocalSizeTuner::decide() {
    const auto fallback_it = std::find(_candidates.begin(), _candidates.end(), _fallback);
    OPENVINO_ASSERT(fallback_it != _candidates.end(), "[GPU][Vulkan] Eltwise local-size candidates lost the portable fallback");
    const auto fallback_index = static_cast<size_t>(std::distance(_candidates.begin(), fallback_it));
    auto best_index = fallback_index;
    auto best_median = median_sample(_samples[fallback_index]);
    for (size_t index = 0; index < _candidates.size(); ++index) {
        const auto candidate_median = median_sample(_samples[index]);
        if (candidate_median < best_median) {
            best_index = index;
            best_median = candidate_median;
        }
    }

    if (best_index != fallback_index) {
        const auto best_max = *std::max_element(_samples[best_index].begin(), _samples[best_index].end());
        const auto fallback_min = *std::min_element(_samples[fallback_index].begin(), _samples[fallback_index].end());
        if (best_max >= fallback_min) {
            best_index = fallback_index;
        }
    }
    _selected = _candidates[best_index];
    _decided = true;
    _cached_selected.store(_selected, std::memory_order_release);

    if (diagnostics_enabled()) {
        std::clog << "[GPU][Vulkan][EltwiseLocalSize] invocations=" << _invocation_count << " fallback=" << _fallback << " selected=" << _selected;
        for (size_t index = 0; index < _candidates.size(); ++index) {
            std::clog << " candidate_" << _candidates[index] << "_median_ns=" << median_sample(_samples[index]);
        }
        std::clog << std::endl;
    }
}

LocalSizeTuner::Action LocalSizeTuner::next_action(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility) {
    if (_invocation_count != invocation_count || _candidates.empty()) {
        initialize(invocation_count, info, require_exact_divisibility);
    }
    if (_decided) {
        return {_selected, 0, false, false};
    }
    if (_fallback_observations < stabilization_inferences) {
        ++_fallback_observations;
        return {_fallback, 0, false, false};
    }
    if (!_prewarmed) {
        return {_fallback, 0, true, false};
    }

    size_t total_samples = 0;
    for (const auto& candidate_samples : _samples) {
        total_samples += candidate_samples.size();
    }
    const auto sample_round = total_samples / _candidates.size();
    if (sample_round == samples_per_candidate) {
        decide();
        return {_selected, 0, false, false};
    }
    const auto position = total_samples % _candidates.size();
    const auto candidate_index = sample_round % 2 == 0 ? position : _candidates.size() - position - 1;
    return {_candidates[candidate_index], candidate_index, false, true};
}

}  // namespace cldnn::vulkan::eltwise_detail
