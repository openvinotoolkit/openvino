// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_local_memory.hpp"

#include <algorithm>
#include <limits>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "openvino/core/except.hpp"

namespace cldnn {

void gpu_local_memory_contract::add(gpu_local_memory_requirement requirement) {
    _requirements.push_back(requirement);
}

bool gpu_local_memory_contract::empty() const noexcept {
    return _requirements.empty();
}

size_t gpu_local_memory_contract::size() const noexcept {
    return _requirements.size();
}

const gpu_local_memory_requirement& gpu_local_memory_contract::operator[](size_t index) const {
    return _requirements.at(index);
}

void gpu_local_memory_contract::validate(const gpu_local_memory_requirement& requirement, uint64_t max_local_memory_bytes) {
    OPENVINO_ASSERT(requirement.byte_size > 0, "[GPU] Local memory requirement cannot be empty");
    OPENVINO_ASSERT(requirement.byte_size <= max_local_memory_bytes,
                    "[GPU] Local memory requirement of ",
                    requirement.byte_size,
                    " bytes exceeds the device limit of ",
                    max_local_memory_bytes,
                    " bytes");
    OPENVINO_ASSERT(requirement.element_size > 0 && (requirement.element_size & (requirement.element_size - 1)) == 0,
                    "[GPU] Local memory element size must be a non-zero power of two");
    OPENVINO_ASSERT(requirement.byte_size % requirement.element_size == 0, "[GPU] Local memory byte size must be divisible by its specialization element size");
    OPENVINO_ASSERT(requirement.mapping != gpu_local_memory_mapping::static_shader || !requirement.runtime_resolved,
                    "[GPU] Runtime-resolved local memory cannot use a static shader mapping");
}

void gpu_local_memory_contract::materialize(kernel_arguments_desc& descriptor,
                                            local_memory_args_desc& backend_arguments,
                                            uint64_t max_local_memory_bytes) const {
    for (const auto& requirement : _requirements) {
        validate(requirement, max_local_memory_bytes);
        switch (requirement.mapping) {
        case gpu_local_memory_mapping::backend_argument: {
            OPENVINO_ASSERT(requirement.byte_size <= std::numeric_limits<size_t>::max(), "[GPU] Local memory requirement exceeds the backend argument range");
            if (backend_arguments.size() <= requirement.mapping_id) {
                backend_arguments.resize(static_cast<size_t>(requirement.mapping_id) + 1);
            }
            backend_arguments[requirement.mapping_id] = static_cast<size_t>(requirement.byte_size);
            const auto argument = std::find_if(descriptor.arguments.begin(), descriptor.arguments.end(), [&](const argument_desc& existing) {
                return existing.t == argument_desc::Types::LOCAL_MEMORY_SIZE && existing.index == requirement.mapping_id;
            });
            if (argument == descriptor.arguments.end()) {
                descriptor.arguments.push_back({argument_desc::Types::LOCAL_MEMORY_SIZE, requirement.mapping_id});
            }
            break;
        }
        case gpu_local_memory_mapping::specialization_constant: {
            OPENVINO_ASSERT(requirement.mapping_id != 0 || !descriptor.specialize_local_size_x,
                            "[GPU] Local memory specialization id 0 conflicts with specialized local size");
            const auto value = requirement.byte_size / requirement.element_size;
            OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU] Local memory specialization value exceeds the 32-bit pipeline range");
            const auto existing = std::find_if(descriptor.specialization_constants.begin(),
                                               descriptor.specialization_constants.end(),
                                               [&](const specialization_constant_desc& constant) {
                                                   return constant.id == requirement.mapping_id;
                                               });
            if (existing == descriptor.specialization_constants.end()) {
                descriptor.specialization_constants.push_back({requirement.mapping_id, static_cast<uint32_t>(value)});
            } else {
                existing->value = static_cast<uint32_t>(value);
            }
            break;
        }
        case gpu_local_memory_mapping::static_shader:
            break;
        }
    }
}

void gpu_local_memory_contract::save(BinaryOutputBuffer& output) const {
    output << current_serialization_version;
    output << _requirements.size();
    for (const auto& requirement : _requirements) {
        output << requirement.byte_size;
        output << requirement.element_size;
        output << static_cast<uint8_t>(requirement.mapping);
        output << requirement.mapping_id;
        output << requirement.runtime_resolved;
    }
}

void gpu_local_memory_contract::load(BinaryInputBuffer& input) {
    uint32_t version = 0;
    input >> version;
    OPENVINO_ASSERT(version == current_serialization_version, "[GPU] Unsupported local memory contract version ", version);

    size_t count = 0;
    input >> count;
    constexpr size_t serialized_requirement_bytes = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint32_t) + sizeof(bool);
    const auto remaining_bytes = input.get_stream_size() - input.get_offset();
    OPENVINO_ASSERT(count <= remaining_bytes / serialized_requirement_bytes, "[GPU] Serialized local memory contract is truncated or corrupt");
    _requirements.clear();
    _requirements.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        gpu_local_memory_requirement requirement;
        uint8_t mapping = 0;
        input >> requirement.byte_size;
        input >> requirement.element_size;
        input >> mapping;
        OPENVINO_ASSERT(mapping <= static_cast<uint8_t>(gpu_local_memory_mapping::static_shader),
                        "[GPU] Invalid serialized local memory mapping ",
                        static_cast<uint32_t>(mapping));
        requirement.mapping = static_cast<gpu_local_memory_mapping>(mapping);
        input >> requirement.mapping_id;
        input >> requirement.runtime_resolved;
        _requirements.push_back(requirement);
    }
}

}  // namespace cldnn
