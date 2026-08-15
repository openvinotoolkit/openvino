// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_stream.hpp"

#include "runtime/except.hpp"
#include "vk_memory.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

namespace {
constexpr uint32_t max_buffer_bindings = 16;
constexpr uint32_t push_constants_size = 128;

size_t scalar_size(const scalar_desc& scalar) {
    switch (scalar.t) {
        case scalar_t::UINT8:
        case scalar_t::INT8:
            return 1;
        case scalar_t::UINT16:
        case scalar_t::INT16:
            return 2;
        case scalar_t::UINT32:
        case scalar_t::INT32:
        case scalar_t::FLOAT32:
            return 4;
        case scalar_t::UINT64:
        case scalar_t::INT64:
        case scalar_t::FLOAT64:
            return 8;
    }
    OPENVINO_ASSERT(false, "[GPU] Unknown scalar type: ", static_cast<int>(scalar.t));
}

uint32_t group_count(size_t global, size_t local) {
    local = std::max<size_t>(local, 1);
    return static_cast<uint32_t>((global + local - 1) / local);
}
}  // namespace

vk_stream::vk_stream(VkDevice device, VkQueue queue, uint32_t queue_family)
    : _device(device)
    , _queue(queue)
    , _queue_family(queue_family) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = _queue_family;
    VK_CALL(vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    buffer_info.commandPool = _command_pool;
    buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    buffer_info.commandBufferCount = 1;
    VK_CALL(vkAllocateCommandBuffers(_device, &buffer_info, &_command_buffer), "vkAllocateCommandBuffers");

    std::vector<VkDescriptorSetLayoutBinding> bindings(max_buffer_bindings);
    for (uint32_t i = 0; i < max_buffer_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    VK_CALL(vkCreateDescriptorSetLayout(_device, &layout_info, nullptr, &_descriptor_set_layout), "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = max_buffer_bindings;

    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;
    VK_CALL(vkCreateDescriptorPool(_device, &descriptor_pool_info, nullptr, &_descriptor_pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = _descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &_descriptor_set_layout;
    VK_CALL(vkAllocateDescriptorSets(_device, &set_info, &_descriptor_set), "vkAllocateDescriptorSets");
}

vk_stream::~vk_stream() {
    if (_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
    }
    if (_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);
    }
    if (_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device, _command_pool, nullptr);
    }
}

void vk_stream::submit_and_wait() const {
    VK_CALL(vkQueueWaitIdle(_queue), "vkQueueWaitIdle");
}

void vk_stream::flush() const {
    submit_and_wait();
}

void vk_stream::finish() const {
    submit_and_wait();
}

void vk_stream::wait() {
    submit_and_wait();
}

vk_event_ptr vk_stream::enqueue_kernel(vk_kernel& kern,
                                       const kernel_arguments_desc& args_desc,
                                       const kernel_arguments_data& args) {
    VK_CALL(vkResetCommandBuffer(_command_buffer, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(_command_buffer, &begin_info), "vkBeginCommandBuffer");

    vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, kern.get_handle());

    std::vector<VkDescriptorBufferInfo> buffer_infos;
    auto add_memory = [&buffer_infos](const std::shared_ptr<const vk_memory>& mem) {
        if (!mem)
            return;
        buffer_infos.push_back({ mem->get_buffer(), 0, VK_WHOLE_SIZE });
    };

    for (auto& mem : args.inputs)
        add_memory(mem);
    for (auto& mem : args.intermediates)
        add_memory(mem);
    for (auto& mem : args.outputs)
        add_memory(mem);
    add_memory(args.weights);
    add_memory(args.recurrent);
    add_memory(args.hidden);
    add_memory(args.cell);
    add_memory(args.bias);
    add_memory(args.weights_zero_points);
    add_memory(args.activations_zero_points);
    add_memory(args.compensation);
    add_memory(args.lookup_table);
    add_memory(args.scale_table);
    add_memory(args.slope);
    add_memory(args.shape_info);
    for (auto& mem : args.fused_op_inputs)
        add_memory(mem);

    OPENVINO_ASSERT(buffer_infos.size() <= max_buffer_bindings,
                    "[GPU] vk_stream::enqueue_kernel: too many arguments (", buffer_infos.size(), ")");

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(buffer_infos.size());
    for (size_t i = 0; i < buffer_infos.size(); ++i) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = _descriptor_set;
        write.dstBinding = static_cast<uint32_t>(i);
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_infos[i];
        writes.push_back(write);
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    VkPipelineLayout pipeline_layout = kern.get_layout();

    vkCmdBindDescriptorSets(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &_descriptor_set, 0, nullptr);

    const auto& reflection = kern.get_reflection();
    const scalars_desc* scalars = args.scalars != nullptr ? args.scalars : &args_desc.scalars;
    const bool has_pod_layout = !reflection.args.empty();
    std::vector<uint8_t> push_data(push_constants_size);
    size_t scalar_index = 0;
    if (has_pod_layout) {
        // Place pod arguments at the offsets reported by the clspv reflection
        // instead of concatenating them sequentially.
        for (const auto& arg : reflection.args) {
            if (arg.kind != vk_arg_kind::pod_push_constant && arg.kind != vk_arg_kind::pointer_push_constant)
                continue;
            if (scalars == nullptr || scalar_index >= scalars->size())
                break;
            const auto& scalar = (*scalars)[scalar_index++];
            const size_t size = scalar_size(scalar);
            OPENVINO_ASSERT(size == arg.size,
                            "[GPU] vk_stream::enqueue_kernel: scalar size mismatch for arg '", arg.name,
                            "' (", size, " != ", arg.size, ")");
            OPENVINO_ASSERT(arg.offset + arg.size <= push_constants_size,
                            "[GPU] vk_stream::enqueue_kernel: push constants overflow for arg '", arg.name, "'");
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&scalar.v);
            std::copy(bytes, bytes + size, push_data.begin() + arg.offset);
        }
    } else if (scalars != nullptr) {
        // Fallback for modules without reflection: concatenate scalars.
        size_t offset = 0;
        for (auto& scalar : *scalars) {
            size_t size = scalar_size(scalar);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&scalar.v);
            OPENVINO_ASSERT(offset + size <= push_constants_size,
                            "[GPU] vk_stream::enqueue_kernel: push constants overflow");
            std::copy(bytes, bytes + size, push_data.begin() + offset);
            offset += size;
        }
    }
    const size_t push_size = has_pod_layout ? reflection.push_constants_size() : push_data.size();
    if (push_size > 0) {
        vkCmdPushConstants(_command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_data.data());
    }

    const work_group_sizes& wgs = args_desc.workGroups;
    // The local size must match the one baked into the shader (LocalSize
    // execution mode) or the driver may reject the dispatch. Modules that use
    // spec-constant workgroup sizes were specialized with reflection.local_size
    // at pipeline creation, so the same values must be used here.
    uint32_t local[3] = {static_cast<uint32_t>(wgs.local[0]), static_cast<uint32_t>(wgs.local[1]), static_cast<uint32_t>(wgs.local[2])};
    if (reflection.has_local_size || reflection.uses_spec_wgsize) {
        local[0] = reflection.local_size[0];
        local[1] = reflection.local_size[1];
        local[2] = reflection.local_size[2];
    }
    vkCmdDispatch(_command_buffer,
                  group_count(wgs.global[0], local[0]),
                  group_count(wgs.global[1], local[1]),
                  group_count(wgs.global[2], local[2]));

    VK_CALL(vkEndCommandBuffer(_command_buffer), "vkEndCommandBuffer");

    auto ev = vk_event::create(_device);
    VkFence fence = ev->get_handle();

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_command_buffer;

    VK_CALL(vkQueueSubmit(_queue, 1, &submit_info, fence), "vkQueueSubmit");
    VK_CALL(vkWaitForFences(_device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    return ev;
}

vk_event_ptr vk_stream::enqueue_marker() {
    return vk_event::create(_device);
}

void vk_stream::enqueue_barrier() {
    submit_and_wait();
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
