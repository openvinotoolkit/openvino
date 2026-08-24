// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "intel_gpu/runtime/device_info.hpp"

namespace cldnn::vulkan::eltwise_detail {

constexpr uint32_t portable_local_work_group_size_limit = 128;

uint32_t select_portable_local_work_group_size(uint32_t invocation_count, uint64_t device_max_work_group_size);

class LocalSizeTuner {
    struct Action {
        uint32_t local_size = 1;
        size_t candidate_index = 0;
        bool prewarm = false;
        bool measure = false;
    };

public:
    class Selection {
    public:
        Selection(const Selection&) = delete;
        Selection& operator=(const Selection&) = delete;
        Selection(Selection&&) noexcept = default;
        Selection& operator=(Selection&&) noexcept = default;

        uint32_t local_size() const;
        bool requires_prewarm() const;
        bool requires_measurement() const;
        const std::vector<uint32_t>& candidates() const;
        uint32_t complete_prewarm();
        void complete_measurement(uint64_t elapsed_nanoseconds);

    private:
        friend class LocalSizeTuner;

        Selection(LocalSizeTuner& owner, Action selected_action, std::unique_lock<std::mutex> lock);

        LocalSizeTuner* _owner;
        Action _action;
        std::unique_lock<std::mutex> _lock;
    };

    uint32_t cached_local_size() const {
        return _cached_selected.load(std::memory_order_acquire);
    }

    Selection select_uncached(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility);

private:
    void initialize(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility);
    void decide();
    Action next_action(uint32_t invocation_count, const device_info& info, bool require_exact_divisibility);

    std::mutex _mutex;
    std::atomic<uint32_t> _cached_selected{0};
    uint32_t _invocation_count = 0;
    uint32_t _fallback = 1;
    uint32_t _selected = 1;
    std::vector<uint32_t> _candidates;
    std::vector<std::vector<uint64_t>> _samples;
    size_t _fallback_observations = 0;
    bool _prewarmed = false;
    bool _decided = false;
};

}  // namespace cldnn::vulkan::eltwise_detail
