// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_stream.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_event.hpp"
#include "vulkan_kernel.hpp"
#include "vulkan_memory.hpp"

namespace cldnn {
namespace vulkan {
namespace {

class empty_surfaces_lock final : public surfaces_lock {};

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

memory::cptr get_argument_memory(const argument_desc& descriptor, const kernel_arguments_data& data) {
    switch (descriptor.t) {
    case argument_desc::Types::INPUT:
        return data.inputs.at(descriptor.index);
    case argument_desc::Types::OUTPUT:
        return data.outputs.at(descriptor.index);
    case argument_desc::Types::WEIGHTS:
        return data.weights;
    case argument_desc::Types::BIAS:
        return data.bias;
    case argument_desc::Types::SCALE_TABLE:
        return data.scale_table;
    case argument_desc::Types::SLOPE:
        return data.slope;
    case argument_desc::Types::INTERNAL_BUFFER:
        return data.intermediates.at(descriptor.index);
    case argument_desc::Types::CELL:
        return data.cell;
    case argument_desc::Types::WEIGHTS_ZERO_POINTS:
        return data.weights_zero_points;
    case argument_desc::Types::ACTIVATIONS_ZERO_POINTS:
        return data.activations_zero_points;
    case argument_desc::Types::COMPENSATION:
        return data.compensation;
    case argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE:
        return data.fused_op_inputs.at(descriptor.index);
    case argument_desc::Types::SHAPE_INFO:
        return data.shape_info;
    case argument_desc::Types::SCALAR:
    case argument_desc::Types::LOCAL_MEMORY_SIZE:
        return nullptr;
    }
    OPENVINO_THROW("[GPU][Vulkan] Unknown kernel argument type");
}

std::vector<uint8_t> pack_push_constants(const scalars_desc& scalars) {
    std::vector<uint8_t> result;
    result.reserve(scalars.size() * sizeof(uint32_t));
    for (const auto& scalar : scalars) {
        OPENVINO_ASSERT(scalar.t == scalar_desc::Types::UINT32 || scalar.t == scalar_desc::Types::INT32 || scalar.t == scalar_desc::Types::FLOAT32,
                        "[GPU][Vulkan] Only 32-bit scalar push constants are currently supported");
        const auto* bytes = reinterpret_cast<const uint8_t*>(&scalar.v.u32);
        result.insert(result.end(), bytes, bytes + sizeof(uint32_t));
    }
    return result;
}

struct dispatch_resources {
    explicit dispatch_resources(VkDevice device) : device(device) {}

    ~dispatch_resources() {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        }
    }

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

struct prepared_arguments {
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<uint8_t> push_constants;
};

prepared_arguments prepare_arguments(const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    prepared_arguments prepared;
    for (const auto& argument : descriptor.arguments) {
        auto memory = get_argument_memory(argument, data);
        if (memory == nullptr) {
            continue;
        }

        const auto* buffer = dynamic_cast<const vulkan_buffer*>(memory.get());
        OPENVINO_ASSERT(buffer != nullptr, "[GPU][Vulkan] Kernel argument is not backed by a Vulkan buffer");
        OPENVINO_ASSERT(buffer->size() > 0, "[GPU][Vulkan] Zero-sized storage buffer arguments are not supported");
        prepared.buffer_infos.push_back({buffer->get_buffer(), buffer->get_offset(), buffer->size()});
    }
    prepared.push_constants = pack_push_constants(descriptor.scalars);
    return prepared;
}

}  // namespace

vulkan_stream::vulkan_stream(const vulkan_engine& engine) : stream(QueueTypes::in_order, SyncMethods::none), _engine(engine) {}

vulkan_stream::vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config)),
      _engine(engine) {}

void vulkan_stream::flush() const {}

void vulkan_stream::finish() const {
    std::lock_guard<std::mutex> lock(_engine.get_queue_mutex());
    const auto result = vkQueueWaitIdle(_engine.get_compute_queue());
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkQueueWaitIdle failed with VkResult ", static_cast<int>(result));
}

void vulkan_stream::wait() {
    finish();
}

void vulkan_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot bind arguments to a kernel from another backend");
    const auto prepared = prepare_arguments(descriptor, data);
    vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()), static_cast<uint32_t>(prepared.push_constants.size()));
}

event::ptr vulkan_stream::enqueue_kernel(kernel& kernel,
                                         const kernel_arguments_desc& descriptor,
                                         const kernel_arguments_data& data,
                                         const std::vector<event::ptr>& dependencies,
                                         bool) {
    wait_for_events(dependencies);

    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot dispatch a kernel from another backend");
    const auto prepared = prepare_arguments(descriptor, data);
    const auto& pipeline =
        vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()), static_cast<uint32_t>(prepared.push_constants.size()));

    const auto device = _engine.get_device_handle();
    dispatch_resources resources(device);

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<uint32_t>(prepared.buffer_infos.size());

    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = prepared.buffer_infos.empty() ? 0 : 1;
    descriptor_pool_info.pPoolSizes = prepared.buffer_infos.empty() ? nullptr : &pool_size;
    check_vk_result(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &resources.descriptor_pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo descriptor_set_info{};
    descriptor_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_info.descriptorPool = resources.descriptor_pool;
    descriptor_set_info.descriptorSetCount = 1;
    descriptor_set_info.pSetLayouts = &pipeline.descriptor_set_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check_vk_result(vkAllocateDescriptorSets(device, &descriptor_set_info, &descriptor_set), "vkAllocateDescriptorSets");

    std::vector<VkWriteDescriptorSet> descriptor_writes(prepared.buffer_infos.size());
    for (uint32_t index = 0; index < descriptor_writes.size(); ++index) {
        descriptor_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[index].dstSet = descriptor_set;
        descriptor_writes[index].dstBinding = index;
        descriptor_writes[index].descriptorCount = 1;
        descriptor_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[index].pBufferInfo = &prepared.buffer_infos[index];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_info.queueFamilyIndex = _engine.get_compute_queue_family();
    check_vk_result(vkCreateCommandPool(device, &command_pool_info, nullptr, &resources.command_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo command_buffer_info{};
    command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_info.commandPool = resources.command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_vk_result(vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk_result(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

    VkMemoryBarrier host_to_shader{};
    host_to_shader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_shader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_to_shader, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    if (!prepared.push_constants.empty()) {
        vkCmdPushConstants(command_buffer,
                           pipeline.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           static_cast<uint32_t>(prepared.push_constants.size()),
                           prepared.push_constants.data());
    }

    OPENVINO_ASSERT(descriptor.workGroups.global.size() == 3 && descriptor.workGroups.local.size() == 3,
                    "[GPU][Vulkan] Compute dispatch requires three-dimensional global and local work-group sizes");
    uint32_t group_counts[3]{};
    for (size_t axis = 0; axis < 3; ++axis) {
        OPENVINO_ASSERT(descriptor.workGroups.local[axis] > 0, "[GPU][Vulkan] Local work-group size cannot be zero");
        group_counts[axis] =
            static_cast<uint32_t>((descriptor.workGroups.global[axis] + descriptor.workGroups.local[axis] - 1) / descriptor.workGroups.local[axis]);
    }
    vkCmdDispatch(command_buffer, group_counts[0], group_counts[1], group_counts[2]);

    VkMemoryBarrier shader_to_host{};
    shader_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    shader_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    shader_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &shader_to_host, 0, nullptr, 0, nullptr);
    check_vk_result(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check_vk_result(vkCreateFence(device, &fence_info, nullptr, &resources.fence), "vkCreateFence");

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    {
        std::lock_guard<std::mutex> lock(_engine.get_queue_mutex());
        check_vk_result(vkQueueSubmit(_engine.get_compute_queue(), 1, &submit_info, resources.fence), "vkQueueSubmit");
    }
    check_vk_result(vkWaitForFences(device, 1, &resources.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    return create_user_event(true);
}

event::ptr vulkan_stream::enqueue_marker(const std::vector<event::ptr>& dependencies, bool) {
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::enqueue_barrier() {
    finish();
}

event::ptr vulkan_stream::group_events(const std::vector<event::ptr>& dependencies) {
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::wait_for_events(const std::vector<event::ptr>& events) {
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

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& vulkan_stream::get_onednn_stream() {
    OPENVINO_THROW("[GPU][Vulkan] oneDNN stream interop is not supported");
}
#endif

}  // namespace vulkan
}  // namespace cldnn
