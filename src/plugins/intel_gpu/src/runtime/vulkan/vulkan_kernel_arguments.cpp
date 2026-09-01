// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_arguments.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace cldnn::vulkan {
namespace {

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

VkAccessFlags2 get_shader_access(argument_desc::Types type) {
    if (type == argument_desc::Types::OUTPUT) {
        return VK_ACCESS_2_SHADER_WRITE_BIT;
    }
    if (type == argument_desc::Types::INTERNAL_BUFFER) {
        return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    }
    return VK_ACCESS_2_SHADER_READ_BIT;
}

}  // namespace

vulkan_prepared_arguments prepare_vulkan_arguments(const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    vulkan_prepared_arguments prepared;
    for (const auto& argument : descriptor.arguments) {
        auto memory = get_argument_memory(argument, data);
        if (memory == nullptr) {
            continue;
        }

        const auto* buffer = dynamic_cast<const vulkan_buffer*>(memory.get());
        OPENVINO_ASSERT(buffer != nullptr, "[GPU][Vulkan] Kernel argument is not backed by a Vulkan buffer");
        OPENVINO_ASSERT(buffer->size() > 0, "[GPU][Vulkan] Zero-sized storage buffer arguments are not supported");
        prepared.buffer_infos.push_back({buffer->get_buffer(), buffer->get_offset(), buffer->size()});
        prepared.allocations.push_back(buffer->get_allocation());
        prepared.accesses.push_back(get_shader_access(argument.t));
        prepared.memories.push_back(std::move(memory));
    }
    prepared.push_constants = pack_push_constants(descriptor.scalars);
    return prepared;
}

}  // namespace cldnn::vulkan
