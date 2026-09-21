// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_buffer_hazard_tracker.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_kernel_arguments.hpp"

namespace cldnn::vulkan {

void vulkan_buffer_hazard_tracker::record(const vulkan_prepared_arguments& prepared, VkPipelineStageFlags2 stages) {
    OPENVINO_ASSERT(prepared.buffer_infos.size() == prepared.allocations.size() && prepared.buffer_infos.size() == prepared.accesses.size(),
                    "[GPU][Vulkan] Prepared buffer access metadata is inconsistent");
    _argument_access_count += prepared.buffer_infos.size();
    discard_expired_accesses();
    _barriers.clear();
    _current_accesses.resize(prepared.buffer_infos.size());
    _access_state_scratch.resize(prepared.buffer_infos.size());
    _has_external_dependency_barrier = false;

    if (_external_dependency_pending) {
        clear_access_history();
        _external_dependency_pending = false;
        _has_external_dependency_barrier = true;
        ++_external_dependency_barrier_count;
    }

    for (size_t index = 0; index < prepared.buffer_infos.size(); ++index) {
        _current_accesses[index] = make_access_range(prepared.buffer_infos[index], stages, prepared.accesses[index], 0);
        const auto state = _access_states.try_emplace(prepared.allocations[index]).first;
        _access_state_scratch[index] = &state->second;
        append_barriers(prepared.allocations[index], state->second.ranges, _current_accesses[index], _barriers);
    }

    _barrier_count += _barriers.size();

    advance_generation();
    for (size_t index = 0; index < _current_accesses.size(); ++index) {
        _current_accesses[index].generation = _generation;
        update_access_state(_access_state_scratch[index]->ranges, _current_accesses[index]);
    }
}

void vulkan_buffer_hazard_tracker::mark_external_dependency() {
    _external_dependency_pending = true;
}

void vulkan_buffer_hazard_tracker::clear() {
    clear_access_history();
    _external_dependency_pending = false;
    _has_external_dependency_barrier = false;
    _barriers.clear();
}

const std::vector<VkBufferMemoryBarrier2>& vulkan_buffer_hazard_tracker::barriers() const {
    return _barriers;
}

bool vulkan_buffer_hazard_tracker::has_external_dependency_barrier() const {
    return _has_external_dependency_barrier;
}

uint64_t vulkan_buffer_hazard_tracker::argument_access_count() const {
    return _argument_access_count;
}

uint64_t vulkan_buffer_hazard_tracker::barrier_count() const {
    return _barrier_count;
}

uint64_t vulkan_buffer_hazard_tracker::external_dependency_barrier_count() const {
    return _external_dependency_barrier_count;
}

const vulkan_buffer_hazard_tracker::access_range* vulkan_buffer_hazard_tracker::range_for_interval(const std::vector<access_range>& ranges,
                                                                                                   VkDeviceSize begin,
                                                                                                   VkDeviceSize end) {
    for (const auto& range : ranges) {
        if (range.begin <= begin && end <= range.end) {
            return &range;
        }
    }
    return nullptr;
}

vulkan_buffer_hazard_tracker::access_range vulkan_buffer_hazard_tracker::make_access_range(const VkDescriptorBufferInfo& info,
                                                                                           VkPipelineStageFlags2 stages,
                                                                                           VkAccessFlags2 access,
                                                                                           uint64_t generation) {
    OPENVINO_ASSERT(info.range > 0 && info.offset <= std::numeric_limits<VkDeviceSize>::max() - info.range,
                    "[GPU][Vulkan] Buffer access range overflows VkDeviceSize");
    return {info.offset, info.offset + info.range, stages, access, generation};
}

void vulkan_buffer_hazard_tracker::update_access_state(std::vector<access_range>& ranges, const access_range& current) {
    if (ranges.empty()) {
        ranges.push_back(current);
        return;
    }
    if (ranges.size() == 1 && ranges.front().begin == current.begin && ranges.front().end == current.end) {
        auto& range = ranges.front();
        range.stages = range.generation == current.generation ? range.stages | current.stages : current.stages;
        range.access = range.generation == current.generation ? range.access | current.access : current.access;
        range.generation = current.generation;
        return;
    }

    const bool overlaps = std::any_of(ranges.begin(), ranges.end(), [&current](const auto& range) {
        return std::max(range.begin, current.begin) < std::min(range.end, current.end);
    });
    if (!overlaps) {
        const auto position = std::lower_bound(ranges.begin(), ranges.end(), current.begin, [](const auto& range, VkDeviceSize begin) {
            return range.begin < begin;
        });
        ranges.insert(position, current);
        return;
    }

    std::vector<VkDeviceSize> boundaries;
    boundaries.reserve(ranges.size() * 2 + 2);
    for (const auto& range : ranges) {
        boundaries.push_back(range.begin);
        boundaries.push_back(range.end);
    }
    boundaries.push_back(current.begin);
    boundaries.push_back(current.end);
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

    std::vector<access_range> result;
    result.reserve(boundaries.size());
    for (size_t index = 1; index < boundaries.size(); ++index) {
        const auto begin = boundaries[index - 1];
        const auto end = boundaries[index];
        const auto* previous = range_for_interval(ranges, begin, end);
        const bool current_covers_interval = current.begin <= begin && end <= current.end;
        if (previous == nullptr && !current_covers_interval) {
            continue;
        }

        auto stages = previous != nullptr ? previous->stages : VkPipelineStageFlags2{0};
        auto access = previous != nullptr ? previous->access : VkAccessFlags2{0};
        auto generation = previous != nullptr ? previous->generation : uint64_t{0};
        if (current_covers_interval) {
            stages = previous != nullptr && previous->generation == current.generation ? previous->stages | current.stages : current.stages;
            access = previous != nullptr && previous->generation == current.generation ? previous->access | current.access : current.access;
            generation = current.generation;
        }
        if (!result.empty() && result.back().end == begin && result.back().stages == stages && result.back().access == access &&
            result.back().generation == generation) {
            result.back().end = end;
        } else {
            result.push_back({begin, end, stages, access, generation});
        }
    }
    ranges = std::move(result);
}

void vulkan_buffer_hazard_tracker::append_barriers(const vulkan_buffer_allocation::ptr& allocation,
                                                   const std::vector<access_range>& previous,
                                                   const access_range& current,
                                                   std::vector<VkBufferMemoryBarrier2>& barriers) {
    for (const auto& previous_range : previous) {
        const auto begin = std::max(previous_range.begin, current.begin);
        const auto end = std::min(previous_range.end, current.end);
        const auto write_access = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        const bool previous_writes = (previous_range.access & write_access) != 0;
        const bool current_writes = (current.access & write_access) != 0;
        if (begin >= end || (!previous_writes && !current_writes)) {
            continue;
        }

        VkBufferMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = previous_range.stages;
        barrier.srcAccessMask = previous_range.access;
        barrier.dstStageMask = current.stages;
        barrier.dstAccessMask = current.access;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = allocation->buffer;
        barrier.offset = begin;
        barrier.size = end - begin;
        const auto duplicate = std::find_if(barriers.begin(), barriers.end(), [&barrier](const auto& existing) {
            return existing.buffer == barrier.buffer && existing.offset == barrier.offset && existing.size == barrier.size &&
                   existing.srcStageMask == barrier.srcStageMask && existing.srcAccessMask == barrier.srcAccessMask &&
                   existing.dstStageMask == barrier.dstStageMask;
        });
        if (duplicate != barriers.end()) {
            duplicate->dstAccessMask |= barrier.dstAccessMask;
        } else {
            barriers.push_back(barrier);
        }
    }
}

void vulkan_buffer_hazard_tracker::clear_access_history() {
    for (auto& entry : _access_states) {
        entry.second.ranges.clear();
    }
}

void vulkan_buffer_hazard_tracker::advance_generation() {
    ++_generation;
    if (_generation != 0) {
        return;
    }
    for (auto& entry : _access_states) {
        for (auto& range : entry.second.ranges) {
            range.generation = 0;
        }
    }
    _generation = 1;
}

void vulkan_buffer_hazard_tracker::discard_expired_accesses() {
    for (auto iterator = _access_states.begin(); iterator != _access_states.end();) {
        if (iterator->first.expired()) {
            iterator = _access_states.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

}  // namespace cldnn::vulkan
