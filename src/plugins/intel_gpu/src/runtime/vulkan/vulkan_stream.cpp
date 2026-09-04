// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_stream.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_buffer_hazard_tracker.hpp"
#include "vulkan_command_sequence_cache.hpp"
#include "vulkan_descriptor_cache.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_event.hpp"
#include "vulkan_kernel.hpp"
#include "vulkan_kernel_arguments.hpp"
#include "vulkan_memory.hpp"

namespace cldnn {
namespace vulkan {
namespace {

class empty_surfaces_lock final : public surfaces_lock {};

constexpr VkDeviceSize buffer_transfer_alignment = sizeof(uint32_t);

bool stream_diagnostics_enabled() {
    const auto* value = std::getenv("OV_GPU_VULKAN_STREAM_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

uint64_t next_stream_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] ", description, " exceeds the 32-bit Vulkan range");
    return static_cast<uint32_t>(value);
}

}  // namespace

struct vulkan_stream::resource_state {
    // These values bound retained resources. They are not
    // selected operating points; the batch limit is chosen per workload below.
    static constexpr size_t max_in_flight_submissions = 8;
    static constexpr size_t max_retained_dispatches_per_batch = 8;
    // This bounds calibration work; measured host costs select the operating path.
    static constexpr size_t max_command_reuse_tuning_samples_per_path = 3;
    // This caps calibration latency; paired inference wins can stop it earlier.
    static constexpr size_t max_pool_reset_tuning_pairs = 7;
    static constexpr size_t max_completion_tuning_pairs = 7;

    struct slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkQueryPool profiling_query_pool = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        uint32_t descriptor_capacity = 0;
        uint32_t descriptor_binding_count = 0;
        VkFence fence = VK_NULL_HANDLE;
        std::shared_ptr<vulkan_submission_state> submission;
        std::vector<memory::cptr> retained_memories;
        std::vector<vulkan_buffer_allocation::ptr> retained_allocations;
        std::vector<std::shared_ptr<const void>> retained_transfer_lifetimes;
        std::vector<std::weak_ptr<vulkan_buffer_allocation>> descriptor_allocations;
        std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
        std::vector<std::shared_ptr<const void>> retained_kernels;
        size_t recorded_dispatches = 0;
        bool transient_command_buffer_submitted = false;
    };

    explicit resource_state(const vulkan_engine& engine, bool enable_profiling = false)
        : device(engine.get_device_handle()),
          descriptor_cache(device),
          command_sequence_cache(device, engine.get_compute_queue_family()),
          queue(engine.get_compute_queue()),
          queue_mutex(engine.get_queue_mutex()),
          stream_id(next_stream_id()),
          max_work_group_invocations(engine.get_device_info().max_work_group_size),
          subgroup_size(engine.get_device_info().supported_simd_sizes.empty() ? 0 : engine.get_device_info().supported_simd_sizes.front()),
          completion_timeline(std::make_shared<vulkan_timeline_state>(device)),
          profiling_enabled(enable_profiling),
          diagnostics_enabled(stream_diagnostics_enabled()) {
        if (profiling_enabled) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(engine.get_physical_device(), &properties);
            OPENVINO_ASSERT(properties.limits.timestampComputeAndGraphics == VK_TRUE && properties.limits.timestampPeriod > 0.0f,
                            "[GPU][Vulkan] The selected device does not support compute timestamp profiling");
            timestamp_period_ns = properties.limits.timestampPeriod;

            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(engine.get_physical_device(), &queue_family_count, nullptr);
            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(engine.get_physical_device(), &queue_family_count, queue_families.data());
            OPENVINO_ASSERT(
                engine.get_compute_queue_family() < queue_families.size() && queue_families[engine.get_compute_queue_family()].timestampValidBits > 0,
                "[GPU][Vulkan] The selected compute queue does not expose valid timestamp bits");
            timestamp_valid_bits = queue_families[engine.get_compute_queue_family()].timestampValidBits;
        }
        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = engine.get_compute_queue_family();
        check_vk_result(vkCreateCommandPool(device, &command_pool_info, nullptr, &transient_command_pool), "vkCreateCommandPool(transient)");
        slots.reserve(max_in_flight_submissions);
    }

    ~resource_state() {
        finish();
        for (auto& slot : slots) {
            recycle(slot);
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, slot.fence, nullptr);
            }
            if (slot.profiling_query_pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device, slot.profiling_query_pool, nullptr);
            }
            if (slot.descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, slot.descriptor_pool, nullptr);
            }
        }
        completion_timeline->close();
        if (transient_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, transient_command_pool, nullptr);
        }
        if (diagnostics_enabled) {
            std::clog << "[GPU][Vulkan][Stream] descriptor_allocations=" << descriptor_allocations << " descriptor_updates=" << descriptor_updates
                      << " descriptor_reuses=" << descriptor_reuses << " dispatches=" << dispatches << " queue_submissions=" << queue_submissions
                      << " max_batch_size=" << largest_batch << " selected_batch_limit_min=" << (dispatches == 0 ? 0 : smallest_selected_batch_limit)
                      << " selected_batch_limit_max=" << largest_selected_batch_limit << " argument_accesses=" << hazard_tracker.argument_access_count()
                      << " hazard_barriers=" << hazard_tracker.barrier_count()
                      << " external_dependency_barriers=" << hazard_tracker.external_dependency_barrier_count()
                      << " command_pool_resets=" << command_pool_resets << " descriptor_pool_resets=" << descriptor_pool_resets
                      << " command_buffer_resets=" << command_buffer_resets << " command_sequence_hits=" << command_sequence_cache.hit_count()
                      << " command_sequence_misses=" << command_sequence_cache.miss_count()
                      << " command_sequences_cached=" << command_sequence_cache.cached_sequence_count()
                      << " last_command_sequence_miss_submission=" << command_sequence_cache.last_miss_submission()
                      << " command_reuse_selected=" << (command_reuse_decided ? (command_reuse_selected ? "yes" : "no") : "calibrating")
                      << " direct_tuning_samples=" << direct_recording_samples.size() << " reuse_tuning_samples=" << command_reuse_samples.size()
                      << " direct_inference_median_ns=" << (direct_recording_samples.empty() ? 0 : median_sample(direct_recording_samples))
                      << " reuse_inference_median_ns=" << (command_reuse_samples.empty() ? 0 : median_sample(command_reuse_samples))
                      << " generation_pool_reset_selected=" << (pool_reset_decided ? (generation_pool_reset_selected ? "yes" : "no") : "calibrating")
                      << " individual_reset_samples=" << individual_reset_samples.size() << " generation_reset_samples=" << generation_reset_samples.size()
                      << " generation_reset_pair_wins=" << generation_reset_pair_wins << " individual_reset_pair_wins=" << individual_reset_pair_wins
                      << " individual_reset_median_ns=" << (individual_reset_samples.empty() ? 0 : median_sample(individual_reset_samples))
                      << " generation_reset_median_ns=" << (generation_reset_samples.empty() ? 0 : median_sample(generation_reset_samples))
                      << " buffer_copies=" << buffer_copies << " buffer_fills=" << buffer_fills << " transfer_bytes=" << transfer_bytes
                      << " completion_timeline_count=1 max_pending_submissions=" << max_pending_submissions
                      << " timeline_completion_selected=" << (completion_tuning_decided ? (timeline_completion_selected ? "yes" : "no") : "calibrating")
                      << " fence_completion_samples=" << fence_completion_samples.size()
                      << " timeline_completion_samples=" << timeline_completion_samples.size()
                      << " fence_completion_median_ns=" << (fence_completion_samples.empty() ? 0 : median_sample(fence_completion_samples))
                      << " timeline_completion_median_ns=" << (timeline_completion_samples.empty() ? 0 : median_sample(timeline_completion_samples))
                      << std::endl;
        }
    }

    slot& get_or_begin_batch(uint32_t descriptor_count,
                             VkDescriptorSetLayout descriptor_set_layout,
                             size_t local_invocations,
                             bool mutable_descriptor_required) {
        const auto selected_limit = select_batch_limit(local_invocations);
        smallest_selected_batch_limit = std::min(smallest_selected_batch_limit, selected_limit);
        largest_selected_batch_limit = std::max(largest_selected_batch_limit, selected_limit);
        if (recording_slot != nullptr) {
            active_batch_limit = std::min(active_batch_limit, selected_limit);
            if (recording_slot->recorded_dispatches < active_batch_limit) {
                return *recording_slot;
            }
            flush();
        }

        auto& slot = acquire(descriptor_count, descriptor_set_layout, mutable_descriptor_required);
        recording_slot = &slot;
        active_batch_limit = selected_limit;
        if (!inference_in_progress) {
            inference_in_progress = true;
            inference_uses_direct_recording = profiling_enabled || select_direct_recording_for_inference();
            inference_all_batches_cache_hits = !inference_uses_direct_recording;
            inference_all_batches_replayable = !inference_uses_direct_recording;
            inference_uses_generation_pool_reset = select_generation_pool_reset_for_inference();
            inference_uses_timeline_completion = select_timeline_completion_for_inference();
            inference_tunes_command_reuse = !profiling_enabled && !command_reuse_decided;
            inference_tunes_pool_reset = !profiling_enabled && command_reuse_decided && !pool_reset_decided;
            inference_tunes_completion = !profiling_enabled && command_reuse_decided && pool_reset_decided && !completion_tuning_decided;
            if (inference_tunes_command_reuse || inference_tunes_pool_reset || inference_tunes_completion) {
                inference_started = std::chrono::steady_clock::now();
            }
        }
        active_submission_uses_timeline = inference_uses_timeline_completion;
        active_direct_recording = inference_uses_direct_recording;
        if (active_direct_recording) {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check_vk_result(vkBeginCommandBuffer(slot.command_buffer, &begin_info), "vkBeginCommandBuffer");
            begin_profiling(slot);
        }
        command_sequence_cache.begin_batch();
        return slot;
    }

    bool descriptor_batch_boundary_required(VkDescriptorSetLayout layout, const vulkan_prepared_arguments& prepared) const {
        return descriptor_cache.requires_mutable_set(layout, prepared);
    }

    void retain_dispatch(slot& slot, const vulkan_prepared_arguments& prepared, std::shared_ptr<const void> kernel_lifetime) {
        OPENVINO_ASSERT(recording_slot == &slot && slot.submission == nullptr, "[GPU][Vulkan] Dispatch resources do not belong to the active command batch");
        slot.retained_memories.insert(slot.retained_memories.end(), prepared.memories.begin(), prepared.memories.end());
        slot.retained_kernels.push_back(std::move(kernel_lifetime));
        ++slot.recorded_dispatches;
        ++dispatches;
    }

    void record_buffer_hazards(slot& slot, const vulkan_prepared_arguments& prepared, VkPipelineStageFlags2 stages) {
        OPENVINO_ASSERT(recording_slot == &slot && slot.submission == nullptr, "[GPU][Vulkan] Buffer hazards must be recorded in the active command batch");
        hazard_tracker.record(prepared, stages);
    }

    void record_dispatch(slot& slot,
                         std::shared_ptr<const vulkan_pipeline_state> pipeline,
                         VkDescriptorSet descriptor_set,
                         vulkan_prepared_arguments& prepared,
                         const std::array<uint32_t, 3>& group_counts,
                         bool descriptor_is_immutable) {
        OPENVINO_ASSERT(recording_slot == &slot && slot.submission == nullptr, "[GPU][Vulkan] Dispatch does not belong to the active command batch");
        command_sequence_cache.append(std::move(pipeline),
                                      descriptor_set,
                                      std::move(prepared.push_constants),
                                      group_counts,
                                      hazard_tracker.barriers(),
                                      hazard_tracker.has_external_dependency_barrier(),
                                      prepared.allocations,
                                      descriptor_is_immutable);
    }

    void record_immediate_dispatch(slot& slot,
                                   const vulkan_pipeline_state& pipeline,
                                   VkDescriptorSet descriptor_set,
                                   const vulkan_prepared_arguments& prepared,
                                   const std::array<uint32_t, 3>& group_counts) const {
        OPENVINO_ASSERT(recording_slot == &slot && active_direct_recording, "[GPU][Vulkan] Immediate dispatch requires an active direct-recording batch");
        const auto& barriers = hazard_tracker.barriers();
        record_vulkan_synchronization(slot.command_buffer, hazard_tracker.has_external_dependency_barrier(), barriers.data(), barriers.size());
        record_vulkan_dispatch(slot.command_buffer, pipeline, descriptor_set, prepared.push_constants, group_counts);
    }

    std::shared_ptr<vulkan_submission_state> submit_buffer_copy(const vulkan_buffer_allocation::ptr& source,
                                                                VkDeviceSize source_offset,
                                                                const vulkan_buffer_allocation::ptr& destination,
                                                                VkDeviceSize destination_offset,
                                                                VkDeviceSize size,
                                                                std::vector<std::shared_ptr<const void>> lifetimes) {
        OPENVINO_ASSERT(source != nullptr && destination != nullptr, "[GPU][Vulkan] Buffer copy allocation is null");
        vulkan_prepared_arguments prepared;
        prepared.buffer_infos = {{source->buffer, source_offset, size}, {destination->buffer, destination_offset, size}};
        prepared.allocations = {source, destination};
        prepared.accesses = {VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
        auto& slot = begin_transfer(prepared);

        VkBufferCopy copy{};
        copy.srcOffset = source_offset;
        copy.dstOffset = destination_offset;
        copy.size = size;
        vkCmdCopyBuffer(slot.command_buffer, source->buffer, destination->buffer, 1, &copy);
        slot.retained_allocations = {source, destination};
        slot.retained_transfer_lifetimes = std::move(lifetimes);
        ++slot.recorded_dispatches;
        ++buffer_copies;
        transfer_bytes += size;
        return flush();
    }

    std::shared_ptr<vulkan_submission_state> submit_buffer_fill(const vulkan_buffer_allocation::ptr& destination,
                                                                VkDeviceSize destination_offset,
                                                                VkDeviceSize size,
                                                                uint32_t pattern,
                                                                std::shared_ptr<const void> lifetime) {
        OPENVINO_ASSERT(destination != nullptr, "[GPU][Vulkan] Buffer fill allocation is null");
        vulkan_prepared_arguments prepared;
        prepared.buffer_infos = {{destination->buffer, destination_offset, size}};
        prepared.allocations = {destination};
        prepared.accesses = {VK_ACCESS_2_TRANSFER_WRITE_BIT};
        auto& slot = begin_transfer(prepared);

        vkCmdFillBuffer(slot.command_buffer, destination->buffer, destination_offset, size, pattern);
        slot.retained_allocations = {destination};
        slot.retained_transfer_lifetimes = {std::move(lifetime)};
        ++slot.recorded_dispatches;
        ++buffer_fills;
        transfer_bytes += size;
        return flush();
    }

    void mark_external_dependency() {
        hazard_tracker.mark_external_dependency();
    }

    bool batch_is_full() const {
        return recording_slot != nullptr && recording_slot->recorded_dispatches >= active_batch_limit;
    }

    bool direct_recording_is_active() const {
        return active_direct_recording;
    }

    std::shared_ptr<vulkan_submission_state> flush() {
        if (recording_slot == nullptr) {
            return latest_submission.lock();
        }

        auto& slot = *recording_slot;
        OPENVINO_ASSERT(slot.recorded_dispatches > 0, "[GPU][Vulkan] Cannot submit an empty command batch");
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        bool cache_hit = false;
        bool sequence_replayable = false;
        if (active_direct_recording) {
            end_profiling(slot);
            check_vk_result(vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");
            slot.transient_command_buffer_submitted = true;
            command_buffer = slot.command_buffer;
        } else {
            OPENVINO_ASSERT(slot.recorded_dispatches == command_sequence_cache.current_dispatch_count(),
                            "[GPU][Vulkan] Recorded dispatch metadata does not match the command batch");
            const auto resolution = command_sequence_cache.resolve(slot.command_buffer, queue_submissions + 1);
            command_buffer = resolution.command_buffer;
            cache_hit = resolution.cache_hit;
            sequence_replayable = resolution.replayable;
            slot.transient_command_buffer_submitted = resolution.uses_transient_buffer;
        }
        if (!active_direct_recording && !cache_hit) {
            inference_all_batches_cache_hits = false;
        }
        if (!active_direct_recording && !sequence_replayable) {
            inference_all_batches_replayable = false;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        const auto completion_value = active_submission_uses_timeline ? next_completion_value++ : 0;
        const VkSemaphore completion_semaphore = completion_timeline->semaphore();
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &completion_value;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        if (active_submission_uses_timeline) {
            submit_info.pNext = &timeline_info;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &completion_semaphore;
        } else {
            OPENVINO_ASSERT(slot.fence != VK_NULL_HANDLE, "[GPU][Vulkan] A fence completion submission has no fence");
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            check_vk_result(vkQueueSubmit(queue, 1, &submit_info, active_submission_uses_timeline ? VK_NULL_HANDLE : slot.fence), "vkQueueSubmit");
        }

        ++queue_submissions;
        largest_batch = std::max(largest_batch, slot.recorded_dispatches);
        recording_slot = nullptr;
        active_batch_limit = 0;
        active_direct_recording = false;
        const auto submitted_with_timeline = active_submission_uses_timeline;
        active_submission_uses_timeline = false;
        command_sequence_cache.begin_batch();
        return mark_submitted(slot, completion_value, submitted_with_timeline);
    }

    void finish() {
        flush();
        if (inference_uses_generation_pool_reset) {
            complete_transient_generation();
        } else {
            complete_transient_commands_individually();
        }
        hazard_tracker.clear();
        command_sequence_cache.discard_expired();
        if (inference_in_progress && inference_tunes_command_reuse) {
            observe_inference_cost(inference_uses_direct_recording,
                                   inference_all_batches_cache_hits,
                                   inference_all_batches_replayable,
                                   std::chrono::steady_clock::now() - inference_started);
        } else if (inference_in_progress && inference_tunes_pool_reset) {
            observe_pool_reset_cost(inference_uses_generation_pool_reset, std::chrono::steady_clock::now() - inference_started);
        } else if (inference_in_progress && inference_tunes_completion) {
            observe_completion_cost(inference_uses_timeline_completion, std::chrono::steady_clock::now() - inference_started);
        }
        inference_in_progress = false;
    }

    VkDescriptorSet get_or_update_descriptor_set(slot& slot,
                                                 const vulkan_pipeline_state& pipeline,
                                                 const vulkan_prepared_arguments& prepared,
                                                 bool& descriptor_is_immutable) {
        const auto descriptor_count = checked_u32(prepared.buffer_infos.size(), "descriptor count");
        OPENVINO_ASSERT(descriptor_count == pipeline.descriptor_count, "[GPU][Vulkan] Descriptor resources do not match the selected pipeline");

        const auto cached = descriptor_cache.lookup_or_allocate(pipeline.descriptor_set_layout, prepared);
        if (cached.status == vulkan_descriptor_cache::lookup_status::reused) {
            descriptor_is_immutable = true;
            ++descriptor_reuses;
            return cached.descriptor_set;
        }
        if (cached.status == vulkan_descriptor_cache::lookup_status::allocated) {
            descriptor_is_immutable = true;
            ++descriptor_allocations;
            ++descriptor_updates;
            return cached.descriptor_set;
        }

        // A cache miss at capacity uses a transient set allocated from the exact
        // descriptor-set layout selected by the pipeline.
        descriptor_is_immutable = false;
        OPENVINO_ASSERT(slot.descriptor_pool != VK_NULL_HANDLE, "[GPU][Vulkan] Mutable descriptor resources were not prepared for the active generation");
        OPENVINO_ASSERT(slot.descriptor_set == VK_NULL_HANDLE ||
                            (slot.descriptor_binding_count == descriptor_count && slot.descriptor_set_layout == pipeline.descriptor_set_layout),
                        "[GPU][Vulkan] Mutable descriptor set is incompatible with the active transient generation");
        if (slot.descriptor_set == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo descriptor_set_info{};
            descriptor_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptor_set_info.descriptorPool = slot.descriptor_pool;
            descriptor_set_info.descriptorSetCount = 1;
            descriptor_set_info.pSetLayouts = &pipeline.descriptor_set_layout;
            check_vk_result(vkAllocateDescriptorSets(device, &descriptor_set_info, &slot.descriptor_set), "vkAllocateDescriptorSets");
            slot.descriptor_set_layout = pipeline.descriptor_set_layout;
            slot.descriptor_binding_count = descriptor_count;
            slot.descriptor_allocations.clear();
            slot.descriptor_buffer_infos.clear();
            ++descriptor_allocations;
        }

        bool bindings_unchanged =
            slot.descriptor_allocations.size() == prepared.allocations.size() && slot.descriptor_buffer_infos.size() == prepared.buffer_infos.size();
        const std::owner_less<std::weak_ptr<vulkan_buffer_allocation>> allocation_less;
        for (size_t index = 0; bindings_unchanged && index < prepared.buffer_infos.size(); ++index) {
            const auto& cached = slot.descriptor_buffer_infos[index];
            const auto& current = prepared.buffer_infos[index];
            const std::weak_ptr<vulkan_buffer_allocation> current_allocation = prepared.allocations[index];
            const auto& cached_allocation = slot.descriptor_allocations[index];
            bindings_unchanged = !cached_allocation.expired() && !allocation_less(cached_allocation, current_allocation) &&
                                 !allocation_less(current_allocation, cached_allocation) && cached.buffer == current.buffer &&
                                 cached.offset == current.offset && cached.range == current.range;
        }
        if (bindings_unchanged) {
            ++descriptor_reuses;
            return slot.descriptor_set;
        }

        descriptor_cache.update(slot.descriptor_set, prepared.buffer_infos);
        slot.descriptor_allocations.clear();
        slot.descriptor_allocations.reserve(prepared.allocations.size());
        for (const auto& allocation : prepared.allocations) {
            slot.descriptor_allocations.emplace_back(allocation);
        }
        slot.descriptor_buffer_infos = prepared.buffer_infos;
        ++descriptor_updates;
        return slot.descriptor_set;
    }

    VkDevice device = VK_NULL_HANDLE;
    vulkan_descriptor_cache descriptor_cache;
    vulkan_command_sequence_cache command_sequence_cache;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex& queue_mutex;
    uint64_t stream_id = 0;
    uint64_t max_work_group_invocations = 0;
    uint64_t subgroup_size = 0;
    VkCommandPool transient_command_pool = VK_NULL_HANDLE;
    std::shared_ptr<vulkan_timeline_state> completion_timeline;
    uint64_t next_completion_value = 1;
    std::vector<slot> slots;
    size_t next_slot = 0;
    bool profiling_enabled = false;
    float timestamp_period_ns = 0.0f;
    uint32_t timestamp_valid_bits = 0;
    bool diagnostics_enabled = false;
    uint64_t descriptor_allocations = 0;
    uint64_t descriptor_updates = 0;
    uint64_t descriptor_reuses = 0;
    uint64_t dispatches = 0;
    uint64_t buffer_copies = 0;
    uint64_t buffer_fills = 0;
    uint64_t transfer_bytes = 0;
    uint64_t queue_submissions = 0;
    uint64_t command_pool_resets = 0;
    uint64_t descriptor_pool_resets = 0;
    uint64_t command_buffer_resets = 0;
    size_t max_pending_submissions = 0;
    size_t largest_batch = 0;
    size_t smallest_selected_batch_limit = max_retained_dispatches_per_batch;
    size_t largest_selected_batch_limit = 0;
    vulkan_buffer_hazard_tracker hazard_tracker;
    std::vector<uint64_t> direct_recording_samples;
    std::vector<uint64_t> command_reuse_samples;
    std::vector<uint64_t> individual_reset_samples;
    std::vector<uint64_t> generation_reset_samples;
    std::vector<uint64_t> fence_completion_samples;
    std::vector<uint64_t> timeline_completion_samples;
    size_t generation_reset_pair_wins = 0;
    size_t individual_reset_pair_wins = 0;
    bool command_reuse_calibration_started = false;
    bool command_reuse_decided = false;
    bool command_reuse_selected = false;
    bool tune_next_inference_with_direct_recording = false;
    bool active_direct_recording = false;
    bool inference_in_progress = false;
    bool inference_uses_direct_recording = false;
    bool inference_all_batches_cache_hits = false;
    bool inference_all_batches_replayable = false;
    bool pool_reset_decided = false;
    bool generation_pool_reset_selected = false;
    bool tune_next_inference_with_generation_pool_reset = false;
    bool inference_uses_generation_pool_reset = false;
    bool inference_tunes_command_reuse = false;
    bool inference_tunes_pool_reset = false;
    bool completion_tuning_decided = false;
    bool timeline_completion_selected = false;
    bool tune_next_inference_with_timeline_completion = false;
    bool inference_uses_timeline_completion = false;
    bool inference_tunes_completion = false;
    bool active_submission_uses_timeline = false;
    std::chrono::steady_clock::time_point inference_started;

private:
    bool select_direct_recording_for_inference() const {
        if (command_reuse_decided) {
            return !command_reuse_selected;
        }
        return command_reuse_calibration_started && tune_next_inference_with_direct_recording;
    }

    bool select_generation_pool_reset_for_inference() const {
        if (!command_reuse_decided) {
            return false;
        }
        if (pool_reset_decided) {
            return generation_pool_reset_selected;
        }
        return tune_next_inference_with_generation_pool_reset;
    }

    bool select_timeline_completion_for_inference() const {
        if (!pool_reset_decided) {
            return false;
        }
        if (completion_tuning_decided) {
            return timeline_completion_selected;
        }
        return tune_next_inference_with_timeline_completion;
    }

    static uint64_t median_sample(std::vector<uint64_t> samples) {
        const auto middle = samples.begin() + samples.size() / 2;
        std::nth_element(samples.begin(), middle, samples.end());
        return *middle;
    }

    void observe_inference_cost(bool used_direct_recording,
                                bool all_batches_cache_hits,
                                bool all_batches_replayable,
                                std::chrono::steady_clock::duration elapsed) {
        if (command_reuse_decided) {
            return;
        }
        if (!used_direct_recording && !all_batches_replayable) {
            command_reuse_selected = false;
            command_reuse_decided = true;
            return;
        }
        const auto elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        if (used_direct_recording) {
            direct_recording_samples.push_back(elapsed_ns);
        } else if (all_batches_cache_hits) {
            command_reuse_samples.push_back(elapsed_ns);
            command_reuse_calibration_started = true;
        } else {
            return;
        }

        if (direct_recording_samples.size() < max_command_reuse_tuning_samples_per_path ||
            command_reuse_samples.size() < max_command_reuse_tuning_samples_per_path) {
            static constexpr std::array<bool, 4> counterbalanced_direct_schedule{false, true, true, false};
            const auto next_sample = direct_recording_samples.size() + command_reuse_samples.size();
            tune_next_inference_with_direct_recording = counterbalanced_direct_schedule[next_sample % counterbalanced_direct_schedule.size()];
            return;
        }
        command_reuse_selected = *std::max_element(command_reuse_samples.begin(), command_reuse_samples.end()) <
                                 *std::min_element(direct_recording_samples.begin(), direct_recording_samples.end());
        command_reuse_decided = true;
    }

    void observe_pool_reset_cost(bool used_generation_pool_reset, std::chrono::steady_clock::duration elapsed) {
        if (pool_reset_decided) {
            return;
        }
        const auto elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        if (used_generation_pool_reset) {
            generation_reset_samples.push_back(elapsed_ns);
        } else {
            individual_reset_samples.push_back(elapsed_ns);
        }

        if (individual_reset_samples.size() == generation_reset_samples.size()) {
            generation_reset_pair_wins = 0;
            individual_reset_pair_wins = 0;
            for (size_t index = 0; index < individual_reset_samples.size(); ++index) {
                if (generation_reset_samples[index] < individual_reset_samples[index]) {
                    ++generation_reset_pair_wins;
                } else {
                    ++individual_reset_pair_wins;
                }
            }
            const auto remaining_pairs = max_pool_reset_tuning_pairs - individual_reset_samples.size();
            const bool generation_is_decisive = generation_reset_pair_wins > individual_reset_pair_wins + remaining_pairs;
            const bool individual_is_decisive = individual_reset_pair_wins > generation_reset_pair_wins + remaining_pairs;
            if (generation_is_decisive || individual_is_decisive || remaining_pairs == 0) {
                generation_pool_reset_selected = generation_reset_pair_wins > individual_reset_pair_wins;
                pool_reset_decided = true;
                return;
            }
        }
        static constexpr std::array<bool, 4> counterbalanced_generation_schedule{false, true, true, false};
        const auto next_sample = individual_reset_samples.size() + generation_reset_samples.size();
        tune_next_inference_with_generation_pool_reset = counterbalanced_generation_schedule[next_sample % counterbalanced_generation_schedule.size()];
    }

    void observe_completion_cost(bool used_timeline, std::chrono::steady_clock::duration elapsed) {
        if (completion_tuning_decided) {
            return;
        }
        const auto elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        (used_timeline ? timeline_completion_samples : fence_completion_samples).push_back(elapsed_ns);
        if (fence_completion_samples.size() == timeline_completion_samples.size()) {
            size_t timeline_wins = 0;
            size_t fence_wins = 0;
            for (size_t index = 0; index < fence_completion_samples.size(); ++index) {
                if (timeline_completion_samples[index] < fence_completion_samples[index]) {
                    ++timeline_wins;
                } else {
                    ++fence_wins;
                }
            }
            const auto remaining_pairs = max_completion_tuning_pairs - fence_completion_samples.size();
            const bool timeline_is_decisive = timeline_wins > fence_wins + remaining_pairs;
            const bool fence_is_decisive = fence_wins > timeline_wins + remaining_pairs;
            if (timeline_is_decisive || fence_is_decisive || remaining_pairs == 0) {
                timeline_completion_selected = timeline_wins > fence_wins;
                completion_tuning_decided = true;
                return;
            }
        }
        static constexpr std::array<bool, 4> counterbalanced_timeline_schedule{false, true, true, false};
        const auto next_sample = fence_completion_samples.size() + timeline_completion_samples.size();
        tune_next_inference_with_timeline_completion = counterbalanced_timeline_schedule[next_sample % counterbalanced_timeline_schedule.size()];
    }

    VkCommandBuffer allocate_command_buffer(VkCommandPool pool) const {
        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        check_vk_result(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");
        return command_buffer;
    }

    size_t select_batch_limit(size_t local_invocations) const {
        const auto usable_local_invocations = std::max<size_t>(local_invocations, 1);
        const auto work_group_capacity = std::max<uint64_t>(max_work_group_invocations / usable_local_invocations, 1);
        if (subgroup_size == 0) {
            return 1;
        }
        const auto subgroup_capacity = std::max<uint64_t>(usable_local_invocations / subgroup_size, 1);
        return std::min<size_t>({work_group_capacity, subgroup_capacity, max_retained_dispatches_per_batch});
    }

    slot& acquire(uint32_t descriptor_count, VkDescriptorSetLayout descriptor_set_layout, bool mutable_descriptor_required) {
        const auto slot_is_compatible = [descriptor_count, descriptor_set_layout, mutable_descriptor_required](const slot& candidate) {
            const bool descriptor_is_compatible =
                !mutable_descriptor_required || candidate.descriptor_set == VK_NULL_HANDLE ||
                (candidate.descriptor_binding_count == descriptor_count && candidate.descriptor_set_layout == descriptor_set_layout);
            return !candidate.transient_command_buffer_submitted && descriptor_is_compatible;
        };
        for (auto& slot : slots) {
            if (slot.submission != nullptr && slot.submission->is_complete()) {
                recycle(slot, !inference_uses_generation_pool_reset);
            }
            if (slot.submission == nullptr && slot_is_compatible(slot)) {
                if (mutable_descriptor_required) {
                    ensure_descriptor_pool(slot, descriptor_count, descriptor_set_layout);
                }
                return slot;
            }
        }

        if (slots.size() < max_in_flight_submissions) {
            slots.emplace_back();
            auto& slot = slots.back();
            slot.command_buffer = allocate_command_buffer(transient_command_pool);

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk_result(vkCreateFence(device, &fence_info, nullptr, &slot.fence), "vkCreateFence");

            if (profiling_enabled) {
                VkQueryPoolCreateInfo query_pool_info{};
                query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                query_pool_info.queryCount = 2;
                check_vk_result(vkCreateQueryPool(device, &query_pool_info, nullptr, &slot.profiling_query_pool), "vkCreateQueryPool(timestamp)");
            }

            if (mutable_descriptor_required) {
                ensure_descriptor_pool(slot, descriptor_count, descriptor_set_layout);
            }
            return slot;
        }

        auto& slot = slots[next_slot++ % slots.size()];
        if (inference_uses_generation_pool_reset) {
            complete_transient_generation();
        } else {
            recycle(slot, true);
        }
        OPENVINO_ASSERT(slot.submission == nullptr && !slot.transient_command_buffer_submitted, "[GPU][Vulkan] Transient command resources were not released");
        if (mutable_descriptor_required) {
            ensure_descriptor_pool(slot, descriptor_count, descriptor_set_layout);
        }
        return slot;
    }

    std::shared_ptr<vulkan_submission_state> mark_submitted(slot& slot, uint64_t completion_value, bool used_timeline) {
        OPENVINO_ASSERT(slot.submission == nullptr && (used_timeline == (completion_value != 0)), "[GPU][Vulkan] Submission resources are already in use");
        const vulkan_profiling_query profiling{slot.profiling_query_pool, timestamp_period_ns, timestamp_valid_bits};
        if (used_timeline) {
            slot.submission = std::make_shared<vulkan_submission_state>(completion_timeline, queue, stream_id, completion_value, profiling);
        } else {
            slot.submission = std::make_shared<vulkan_submission_state>(device, queue, slot.fence, stream_id, profiling);
            slot.fence = VK_NULL_HANDLE;
        }
        const auto pending_submissions = std::count_if(slots.begin(), slots.end(), [](const auto& candidate) {
            return candidate.submission != nullptr;
        });
        max_pending_submissions = std::max(max_pending_submissions, static_cast<size_t>(pending_submissions));
        latest_submission = slot.submission;
        return slot.submission;
    }
    void recycle(slot& slot, bool reset_transient_command = true) {
        if (slot.submission != nullptr) {
            slot.submission->wait();
            slot.submission->capture_profiling();
            if (slot.submission->uses_fence()) {
                slot.fence = slot.submission->release_fence();
                check_vk_result(vkResetFences(device, 1, &slot.fence), "vkResetFences");
            }
            slot.submission.reset();
            slot.retained_memories.clear();
            slot.retained_allocations.clear();
            slot.retained_transfer_lifetimes.clear();
            slot.retained_kernels.clear();
            slot.recorded_dispatches = 0;
        }

        // A generation recycle can release the submission while deliberately leaving its
        // transient command buffer pending for a pool reset. If tuning switches to individual
        // reset before that pool reset, the slot still has to be reset even without a submission.
        if (reset_transient_command && slot.transient_command_buffer_submitted) {
            check_vk_result(vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
            ++command_buffer_resets;
            slot.transient_command_buffer_submitted = false;
        }
    }

    void complete_transient_commands_individually() {
        for (auto& slot : slots) {
            recycle(slot, true);
        }
    }

    void complete_transient_generation() {
        for (auto& slot : slots) {
            recycle(slot, false);
        }

        const bool has_transient_commands = std::any_of(slots.begin(), slots.end(), [](const auto& slot) {
            return slot.transient_command_buffer_submitted;
        });
        if (has_transient_commands) {
            check_vk_result(vkResetCommandPool(device, transient_command_pool, 0), "vkResetCommandPool(transient generation)");
            ++command_pool_resets;
            for (auto& slot : slots) {
                slot.transient_command_buffer_submitted = false;
            }
        }

        for (auto& slot : slots) {
            if (slot.descriptor_set == VK_NULL_HANDLE) {
                continue;
            }
            check_vk_result(vkResetDescriptorPool(device, slot.descriptor_pool, 0), "vkResetDescriptorPool(transient generation)");
            ++descriptor_pool_resets;
            slot.descriptor_set = VK_NULL_HANDLE;
            slot.descriptor_set_layout = VK_NULL_HANDLE;
            slot.descriptor_binding_count = 0;
            slot.descriptor_allocations.clear();
            slot.descriptor_buffer_infos.clear();
        }
    }

    void ensure_descriptor_pool(slot& slot, uint32_t descriptor_count, VkDescriptorSetLayout descriptor_set_layout) {
        if (slot.descriptor_pool != VK_NULL_HANDLE && slot.descriptor_capacity >= descriptor_count) {
            if (slot.descriptor_set == VK_NULL_HANDLE || slot.descriptor_set_layout == descriptor_set_layout) {
                return;
            }
            check_vk_result(vkResetDescriptorPool(device, slot.descriptor_pool, 0), "vkResetDescriptorPool(incompatible layout)");
            ++descriptor_pool_resets;
        } else if (slot.descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, slot.descriptor_pool, nullptr);
            slot.descriptor_pool = VK_NULL_HANDLE;
        }
        slot.descriptor_set = VK_NULL_HANDLE;
        slot.descriptor_set_layout = VK_NULL_HANDLE;
        slot.descriptor_binding_count = 0;
        slot.descriptor_allocations.clear();
        slot.descriptor_buffer_infos.clear();

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = descriptor_count;

        VkDescriptorPoolCreateInfo descriptor_pool_info{};
        descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_info.maxSets = 1;
        descriptor_pool_info.poolSizeCount = descriptor_count == 0 ? 0 : 1;
        descriptor_pool_info.pPoolSizes = descriptor_count == 0 ? nullptr : &pool_size;
        if (slot.descriptor_pool == VK_NULL_HANDLE) {
            check_vk_result(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &slot.descriptor_pool), "vkCreateDescriptorPool");
            slot.descriptor_capacity = descriptor_count;
        }
    }

    slot* recording_slot = nullptr;
    size_t active_batch_limit = 0;
    std::weak_ptr<vulkan_submission_state> latest_submission;

    void begin_profiling(const slot& slot) const {
        if (!profiling_enabled) {
            return;
        }
        OPENVINO_ASSERT(slot.profiling_query_pool != VK_NULL_HANDLE, "[GPU][Vulkan] Profiling-enabled command buffer has no timestamp query pool");
        vkCmdResetQueryPool(slot.command_buffer, slot.profiling_query_pool, 0, 2);
        vkCmdWriteTimestamp2(slot.command_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, slot.profiling_query_pool, 0);
    }

    void end_profiling(const slot& slot) const {
        if (!profiling_enabled) {
            return;
        }
        vkCmdWriteTimestamp2(slot.command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, slot.profiling_query_pool, 1);
    }

    slot& begin_transfer(const vulkan_prepared_arguments& prepared) {
        flush();
        auto& slot = acquire(0, VK_NULL_HANDLE, false);
        recording_slot = &slot;
        active_batch_limit = 1;
        active_direct_recording = true;
        active_submission_uses_timeline = completion_tuning_decided && timeline_completion_selected;
        command_sequence_cache.begin_batch();

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk_result(vkBeginCommandBuffer(slot.command_buffer, &begin_info), "vkBeginCommandBuffer(transfer)");
        begin_profiling(slot);
        record_buffer_hazards(slot, prepared, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
        const auto& barriers = hazard_tracker.barriers();
        record_vulkan_synchronization(slot.command_buffer, hazard_tracker.has_external_dependency_barrier(), barriers.data(), barriers.size());
        return slot;
    }
};

vulkan_stream::vulkan_stream(const vulkan_engine& engine)
    : stream(QueueTypes::in_order, SyncMethods::none),
      _engine(engine),
      _resources(std::make_unique<resource_state>(engine)) {}

vulkan_stream::vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config)),
      _engine(engine),
      _resources(std::make_unique<resource_state>(engine, config.get_enable_profiling())) {}

vulkan_stream::~vulkan_stream() = default;

void vulkan_stream::flush() const {
    _resources->flush();
}

void vulkan_stream::finish() const {
    _resources->finish();
}

void vulkan_stream::wait() {
    finish();
}

void vulkan_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot bind arguments to a kernel from another backend");
    const auto prepared = prepare_vulkan_arguments(descriptor, data);
    OPENVINO_ASSERT(descriptor.workGroups.local.size() == 3, "[GPU][Vulkan] Compute dispatch requires a three-dimensional local work-group size");
    const std::array<size_t, 3> local_size{descriptor.workGroups.local[0], descriptor.workGroups.local[1], descriptor.workGroups.local[2]};
    vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()), static_cast<uint32_t>(prepared.push_constants.size()), local_size);
}

event::ptr vulkan_stream::enqueue_kernel(kernel& kernel,
                                         const kernel_arguments_desc& descriptor,
                                         const kernel_arguments_data& data,
                                         const std::vector<event::ptr>& dependencies,
                                         bool is_output_event) {
    return enqueue_kernel(kernel, descriptor, data, {}, dependencies, is_output_event);
}

event::ptr vulkan_stream::enqueue_kernel(kernel& kernel,
                                         const kernel_arguments_desc& descriptor,
                                         const kernel_arguments_data& data,
                                         const vulkan_specialization_constants& specialization_constants,
                                         const std::vector<event::ptr>& dependencies,
                                         bool is_output_event) {
    for (const auto& dependency : dependencies) {
        if (dependency == nullptr) {
            continue;
        }
        const auto* vk_event = dynamic_cast<const vulkan_event*>(dependency.get());
        if (vk_event == nullptr || !vk_event->is_device_submission(_engine.get_device_handle(), _engine.get_compute_queue())) {
            _resources->flush();
            dependency->wait();
        } else if (!vk_event->is_stream_submission(_engine.get_device_handle(), _engine.get_compute_queue(), _resources->stream_id)) {
            _resources->flush();
            _resources->mark_external_dependency();
        }
    }

    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot dispatch a kernel from another backend");
    auto prepared = prepare_vulkan_arguments(descriptor, data);
    OPENVINO_ASSERT(descriptor.workGroups.global.size() == 3 && descriptor.workGroups.local.size() == 3,
                    "[GPU][Vulkan] Compute dispatch requires three-dimensional global and local work-group sizes");
    std::array<uint32_t, 3> group_counts{};
    std::array<size_t, 3> local_sizes{};
    size_t local_invocations = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        const auto local_size = descriptor.workGroups.local[axis];
        OPENVINO_ASSERT(local_size > 0, "[GPU][Vulkan] Local work-group size cannot be zero");
        OPENVINO_ASSERT(local_invocations <= std::numeric_limits<size_t>::max() / local_size,
                        "[GPU][Vulkan] Local work-group invocation count overflows size_t");
        local_invocations *= local_size;
        local_sizes[axis] = local_size;
        group_counts[axis] = static_cast<uint32_t>((descriptor.workGroups.global[axis] + local_size - 1) / local_size);
    }
    const auto pipeline = vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()),
                                                            static_cast<uint32_t>(prepared.push_constants.size()),
                                                            local_sizes,
                                                            specialization_constants);

    const bool mutable_descriptor_required = _resources->descriptor_batch_boundary_required(pipeline->descriptor_set_layout, prepared);
    if (mutable_descriptor_required) {
        _resources->flush();
    }
    auto& resources = _resources->get_or_begin_batch(checked_u32(prepared.buffer_infos.size(), "descriptor count"),
                                                     pipeline->descriptor_set_layout,
                                                     local_invocations,
                                                     mutable_descriptor_required);

    bool descriptor_is_immutable = false;
    const VkDescriptorSet descriptor_set = _resources->get_or_update_descriptor_set(resources, *pipeline, prepared, descriptor_is_immutable);
    _resources->record_buffer_hazards(resources, prepared, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    if (_resources->direct_recording_is_active()) {
        _resources->record_immediate_dispatch(resources, *pipeline, descriptor_set, prepared, group_counts);
    } else {
        _resources->record_dispatch(resources, pipeline, descriptor_set, prepared, group_counts, descriptor_is_immutable);
    }
    _resources->retain_dispatch(resources, prepared, pipeline);
    if (is_output_event || m_sync_method == SyncMethods::events || _resources->batch_is_full()) {
        return std::make_shared<vulkan_event>(_resources->flush());
    }
    return nullptr;
}

event::ptr vulkan_stream::enqueue_marker(const std::vector<event::ptr>& dependencies, bool) {
    _resources->flush();
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::enqueue_barrier() {
    finish();
}

event::ptr vulkan_stream::group_events(const std::vector<event::ptr>& dependencies) {
    if (dependencies.size() == 1) {
        return dependencies.front();
    }
    return std::make_shared<vulkan_events>(dependencies);
}

void vulkan_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (!events.empty()) {
        _resources->flush();
    }
    for (const auto& event : events) {
        if (event != nullptr) {
            event->wait();
        }
    }
}

event::ptr vulkan_stream::create_user_event(bool set) {
    return std::make_shared<vulkan_event>(set);
}

event::ptr vulkan_stream::create_base_event() {
    return std::make_shared<vulkan_event>();
}

std::unique_ptr<surfaces_lock> vulkan_stream::create_surfaces_lock(const std::vector<memory::ptr>&) const {
    return std::make_unique<empty_surfaces_lock>();
}

event::ptr vulkan_stream::enqueue_buffer_copy(const std::shared_ptr<vulkan_buffer_allocation>& source,
                                              VkDeviceSize source_offset,
                                              const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                              VkDeviceSize destination_offset,
                                              VkDeviceSize size,
                                              bool blocking,
                                              std::vector<std::shared_ptr<const void>> lifetimes) const {
    OPENVINO_ASSERT(source != nullptr && destination != nullptr, "[GPU][Vulkan] Buffer copy allocation is null");
    OPENVINO_ASSERT(source_offset <= source->size && size <= source->size - source_offset, "[GPU][Vulkan] Buffer copy source range exceeds its allocation");
    OPENVINO_ASSERT(destination_offset <= destination->size && size <= destination->size - destination_offset,
                    "[GPU][Vulkan] Buffer copy destination range exceeds its allocation");
    if (size == 0) {
        return std::make_shared<vulkan_event>(true);
    }
    OPENVINO_ASSERT(
        source_offset % buffer_transfer_alignment == 0 && destination_offset % buffer_transfer_alignment == 0 && size % buffer_transfer_alignment == 0,
        "[GPU][Vulkan] Buffer copy range is not transfer-aligned");
    if (source == destination) {
        const auto non_overlapping =
            source_offset < destination_offset ? size <= destination_offset - source_offset : size <= source_offset - destination_offset;
        OPENVINO_ASSERT(non_overlapping, "[GPU][Vulkan] Buffer copy ranges overlap in one allocation");
    }

    auto event =
        std::make_shared<vulkan_event>(_resources->submit_buffer_copy(source, source_offset, destination, destination_offset, size, std::move(lifetimes)));
    if (blocking) {
        event->wait();
    }
    return event;
}

event::ptr vulkan_stream::enqueue_buffer_fill(const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                              VkDeviceSize destination_offset,
                                              VkDeviceSize size,
                                              uint32_t pattern,
                                              std::shared_ptr<const void> lifetime,
                                              const std::vector<event::ptr>& dependencies,
                                              bool blocking) const {
    OPENVINO_ASSERT(destination != nullptr, "[GPU][Vulkan] Buffer fill allocation is null");
    OPENVINO_ASSERT(destination_offset <= destination->size && size <= destination->size - destination_offset,
                    "[GPU][Vulkan] Buffer fill range exceeds its allocation");
    if (size == 0) {
        return std::make_shared<vulkan_event>(true);
    }
    OPENVINO_ASSERT(destination_offset % buffer_transfer_alignment == 0 && size % buffer_transfer_alignment == 0,
                    "[GPU][Vulkan] Buffer fill range is not transfer-aligned");

    for (const auto& dependency : dependencies) {
        if (dependency == nullptr) {
            continue;
        }
        const auto* vk_event = dynamic_cast<const vulkan_event*>(dependency.get());
        if (vk_event == nullptr || !vk_event->is_device_submission(_engine.get_device_handle(), _engine.get_compute_queue())) {
            _resources->flush();
            dependency->wait();
        } else if (!vk_event->is_stream_submission(_engine.get_device_handle(), _engine.get_compute_queue(), _resources->stream_id)) {
            _resources->flush();
            _resources->mark_external_dependency();
        }
    }

    auto event = std::make_shared<vulkan_event>(_resources->submit_buffer_fill(destination, destination_offset, size, pattern, std::move(lifetime)));
    if (blocking) {
        event->wait();
    }
    return event;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& vulkan_stream::get_onednn_stream() {
    OPENVINO_THROW("[GPU][Vulkan] oneDNN stream interop is not supported");
}
#endif

}  // namespace vulkan
}  // namespace cldnn
