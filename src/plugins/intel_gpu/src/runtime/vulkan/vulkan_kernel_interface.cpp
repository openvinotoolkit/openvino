// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_interface.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include "openvino/core/except.hpp"

namespace cldnn::vulkan {
namespace {

constexpr uint32_t spirv_magic = 0x07230203;

enum class spirv_opcode : uint16_t {
    entry_point = 15,
    execution_mode = 16,
    execution_mode_id = 331,
    type_bool = 20,
    type_int = 21,
    type_float = 22,
    type_vector = 23,
    type_matrix = 24,
    type_array = 28,
    type_runtime_array = 29,
    type_struct = 30,
    type_pointer = 32,
    constant = 43,
    spec_constant = 50,
    spec_constant_composite = 51,
    variable = 59,
    decorate = 71,
    member_decorate = 72,
};

enum class spirv_decoration : uint32_t {
    spec_id = 1,
    block = 2,
    buffer_block = 3,
    array_stride = 6,
    matrix_stride = 7,
    built_in = 11,
    binding = 33,
    descriptor_set = 34,
    offset = 35,
};

enum class spirv_storage_class : uint32_t {
    uniform = 2,
    push_constant = 9,
    storage_buffer = 12,
};

enum class spirv_execution_mode : uint32_t {
    local_size = 17,
    local_size_id = 38,
};

enum class spirv_builtin : uint32_t {
    workgroup_size = 25,
};

struct decoration_set final {
    std::optional<uint32_t> descriptor_set;
    std::optional<uint32_t> binding;
    std::optional<uint32_t> spec_id;
    std::optional<uint32_t> array_stride;
    std::optional<uint32_t> built_in;
    bool block = false;
    bool buffer_block = false;
};

struct type_description final {
    enum class kind {
        boolean,
        integer,
        floating_point,
        vector,
        matrix,
        array,
        runtime_array,
        structure,
        pointer,
    };

    kind type_kind = kind::boolean;
    uint32_t width = 0;
    uint32_t element_type = 0;
    uint32_t element_count = 0;
    uint32_t length_id = 0;
    uint32_t storage_class = 0;
    std::vector<uint32_t> member_types;
};

struct variable_description final {
    uint32_t result_type = 0;
    uint32_t result_id = 0;
    uint32_t storage_class = 0;
};

struct reflection_state final {
    std::map<uint32_t, decoration_set> decorations;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> member_offsets;
    std::map<uint32_t, type_description> types;
    std::map<uint32_t, uint64_t> constants;
    std::map<uint32_t, std::vector<uint32_t>> spec_constant_composites;
    std::map<uint32_t, std::array<uint32_t, 3>> entry_point_local_size_literals;
    std::map<uint32_t, std::array<uint32_t, 3>> entry_point_local_size_ids;
    std::vector<variable_description> variables;
    std::optional<uint32_t> entry_point_id;
    std::set<uint32_t> entry_point_interface_ids;
};

std::string decode_spirv_string(const uint32_t* words, size_t word_count) {
    const auto* bytes = reinterpret_cast<const char*>(words);
    const auto byte_count = word_count * sizeof(uint32_t);
    size_t length = 0;
    while (length < byte_count && bytes[length] != '\0') {
        ++length;
    }
    OPENVINO_ASSERT(length < byte_count, "[GPU][Vulkan] SPIR-V contains an unterminated entry-point name");
    return std::string(bytes, length);
}

size_t spirv_string_word_count(const uint32_t* words, size_t word_count) {
    const auto* bytes = reinterpret_cast<const char*>(words);
    const auto byte_count = word_count * sizeof(uint32_t);
    for (size_t index = 0; index < byte_count; ++index) {
        if (bytes[index] == '\0') {
            return index / sizeof(uint32_t) + 1;
        }
    }
    OPENVINO_THROW("[GPU][Vulkan] SPIR-V contains an unterminated entry-point name");
}

uint32_t checked_u32(uint64_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] Reflected ", description, " exceeds the 32-bit range");
    return static_cast<uint32_t>(value);
}

uint32_t reflected_type_size(uint32_t type_id, const reflection_state& state, std::set<uint32_t>& active_types) {
    const auto type_iterator = state.types.find(type_id);
    OPENVINO_ASSERT(type_iterator != state.types.end(), "[GPU][Vulkan] SPIR-V interface references unknown type id ", type_id);
    OPENVINO_ASSERT(active_types.insert(type_id).second, "[GPU][Vulkan] SPIR-V interface contains a recursive value type");

    const auto& type = type_iterator->second;
    uint64_t size = 0;
    switch (type.type_kind) {
    case type_description::kind::boolean:
        size = sizeof(uint32_t);
        break;
    case type_description::kind::integer:
    case type_description::kind::floating_point:
        OPENVINO_ASSERT(type.width != 0 && type.width % 8 == 0, "[GPU][Vulkan] SPIR-V scalar type has an invalid bit width");
        size = type.width / 8;
        break;
    case type_description::kind::vector:
        size = static_cast<uint64_t>(reflected_type_size(type.element_type, state, active_types)) * type.element_count;
        break;
    case type_description::kind::matrix:
        OPENVINO_THROW("[GPU][Vulkan] Matrix push constants are not supported by the canonical compute ABI");
    case type_description::kind::array: {
        const auto length_iterator = state.constants.find(type.length_id);
        OPENVINO_ASSERT(length_iterator != state.constants.end(), "[GPU][Vulkan] SPIR-V interface array length is not a constant");
        const auto decoration_iterator = state.decorations.find(type_id);
        const auto element_size = reflected_type_size(type.element_type, state, active_types);
        auto stride = element_size;
        if (decoration_iterator != state.decorations.end() && decoration_iterator->second.array_stride.has_value()) {
            stride = *decoration_iterator->second.array_stride;
        }
        if (length_iterator->second != 0) {
            size = static_cast<uint64_t>(stride) * (length_iterator->second - 1) + element_size;
        }
        break;
    }
    case type_description::kind::structure:
        for (uint32_t member = 0; member < type.member_types.size(); ++member) {
            const auto offset_iterator = state.member_offsets.find({type_id, member});
            OPENVINO_ASSERT(offset_iterator != state.member_offsets.end(), "[GPU][Vulkan] SPIR-V push-constant member is missing an Offset decoration");
            const auto member_size = reflected_type_size(type.member_types[member], state, active_types);
            size = std::max<uint64_t>(size, static_cast<uint64_t>(offset_iterator->second) + member_size);
        }
        break;
    case type_description::kind::pointer:
        size = reflected_type_size(type.element_type, state, active_types);
        break;
    case type_description::kind::runtime_array:
        OPENVINO_THROW("[GPU][Vulkan] Runtime arrays are not valid in the canonical push-constant ABI");
    }

    active_types.erase(type_id);
    return checked_u32(size, "type size");
}

reflection_state parse_spirv(const std::vector<uint8_t>& spirv, const std::string& requested_entry_point) {
    OPENVINO_ASSERT(spirv.size() >= 5 * sizeof(uint32_t) && spirv.size() % sizeof(uint32_t) == 0, "[GPU][Vulkan] SPIR-V binary has an invalid size");
    std::vector<uint32_t> words(spirv.size() / sizeof(uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    OPENVINO_ASSERT(words.front() == spirv_magic, "[GPU][Vulkan] Invalid SPIR-V magic number");

    reflection_state state;
    for (size_t offset = 5; offset < words.size();) {
        const auto instruction = words[offset];
        const auto word_count = static_cast<uint16_t>(instruction >> 16);
        const auto opcode = static_cast<spirv_opcode>(instruction & 0xffffU);
        OPENVINO_ASSERT(word_count > 0 && word_count <= words.size() - offset, "[GPU][Vulkan] SPIR-V contains a truncated instruction");
        const auto* operands = words.data() + offset + 1;
        const auto operand_count = static_cast<size_t>(word_count - 1);

        switch (opcode) {
        case spirv_opcode::entry_point:
            OPENVINO_ASSERT(operand_count >= 3, "[GPU][Vulkan] Invalid OpEntryPoint instruction");
            if (decode_spirv_string(operands + 2, operand_count - 2) == requested_entry_point) {
                OPENVINO_ASSERT(!state.entry_point_id.has_value(), "[GPU][Vulkan] SPIR-V contains duplicate entry point ", requested_entry_point);
                state.entry_point_id = operands[1];
                const auto name_word_count = spirv_string_word_count(operands + 2, operand_count - 2);
                state.entry_point_interface_ids.insert(operands + 2 + name_word_count, operands + operand_count);
            }
            break;
        case spirv_opcode::execution_mode:
            OPENVINO_ASSERT(operand_count >= 2, "[GPU][Vulkan] Invalid OpExecutionMode instruction");
            if (static_cast<spirv_execution_mode>(operands[1]) == spirv_execution_mode::local_size) {
                OPENVINO_ASSERT(operand_count == 5, "[GPU][Vulkan] Invalid LocalSize execution mode");
                std::array<uint32_t, 3> local_size{};
                std::copy_n(operands + 2, local_size.size(), local_size.begin());
                state.entry_point_local_size_literals[operands[0]] = local_size;
            }
            break;
        case spirv_opcode::execution_mode_id:
            OPENVINO_ASSERT(operand_count >= 2, "[GPU][Vulkan] Invalid OpExecutionModeId instruction");
            if (static_cast<spirv_execution_mode>(operands[1]) == spirv_execution_mode::local_size_id) {
                OPENVINO_ASSERT(operand_count == 5, "[GPU][Vulkan] Invalid LocalSizeId execution mode");
                std::array<uint32_t, 3> local_size_ids{};
                std::copy_n(operands + 2, local_size_ids.size(), local_size_ids.begin());
                state.entry_point_local_size_ids[operands[0]] = local_size_ids;
            }
            break;
        case spirv_opcode::decorate: {
            OPENVINO_ASSERT(operand_count >= 2, "[GPU][Vulkan] Invalid OpDecorate instruction");
            auto& decoration = state.decorations[operands[0]];
            switch (static_cast<spirv_decoration>(operands[1])) {
            case spirv_decoration::spec_id:
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid SpecId decoration");
                decoration.spec_id = operands[2];
                break;
            case spirv_decoration::block:
                decoration.block = true;
                break;
            case spirv_decoration::buffer_block:
                decoration.buffer_block = true;
                break;
            case spirv_decoration::array_stride:
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid ArrayStride decoration");
                decoration.array_stride = operands[2];
                break;
            case spirv_decoration::built_in:
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid BuiltIn decoration");
                decoration.built_in = operands[2];
                break;
            case spirv_decoration::binding:
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid Binding decoration");
                decoration.binding = operands[2];
                break;
            case spirv_decoration::descriptor_set:
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid DescriptorSet decoration");
                decoration.descriptor_set = operands[2];
                break;
            default:
                break;
            }
            break;
        }
        case spirv_opcode::member_decorate:
            OPENVINO_ASSERT(operand_count >= 3, "[GPU][Vulkan] Invalid OpMemberDecorate instruction");
            if (static_cast<spirv_decoration>(operands[2]) == spirv_decoration::offset) {
                OPENVINO_ASSERT(operand_count == 4, "[GPU][Vulkan] Invalid member Offset decoration");
                state.member_offsets[{operands[0], operands[1]}] = operands[3];
            }
            break;
        case spirv_opcode::type_bool:
            OPENVINO_ASSERT(operand_count == 1, "[GPU][Vulkan] Invalid OpTypeBool instruction");
            state.types[operands[0]].type_kind = type_description::kind::boolean;
            break;
        case spirv_opcode::type_int:
        case spirv_opcode::type_float: {
            OPENVINO_ASSERT(operand_count >= 2, "[GPU][Vulkan] Invalid scalar type instruction");
            auto& type = state.types[operands[0]];
            type.type_kind = opcode == spirv_opcode::type_int ? type_description::kind::integer : type_description::kind::floating_point;
            type.width = operands[1];
            break;
        }
        case spirv_opcode::type_vector:
        case spirv_opcode::type_matrix: {
            OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid vector or matrix type instruction");
            auto& type = state.types[operands[0]];
            type.type_kind = opcode == spirv_opcode::type_vector ? type_description::kind::vector : type_description::kind::matrix;
            type.element_type = operands[1];
            type.element_count = operands[2];
            break;
        }
        case spirv_opcode::type_array:
        case spirv_opcode::type_runtime_array: {
            OPENVINO_ASSERT(operand_count >= 2, "[GPU][Vulkan] Invalid array type instruction");
            auto& type = state.types[operands[0]];
            type.type_kind = opcode == spirv_opcode::type_array ? type_description::kind::array : type_description::kind::runtime_array;
            type.element_type = operands[1];
            if (opcode == spirv_opcode::type_array) {
                OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid OpTypeArray instruction");
                type.length_id = operands[2];
            }
            break;
        }
        case spirv_opcode::type_struct: {
            OPENVINO_ASSERT(operand_count >= 1, "[GPU][Vulkan] Invalid OpTypeStruct instruction");
            auto& type = state.types[operands[0]];
            type.type_kind = type_description::kind::structure;
            type.member_types.assign(operands + 1, operands + operand_count);
            break;
        }
        case spirv_opcode::type_pointer: {
            OPENVINO_ASSERT(operand_count == 3, "[GPU][Vulkan] Invalid OpTypePointer instruction");
            auto& type = state.types[operands[0]];
            type.type_kind = type_description::kind::pointer;
            type.storage_class = operands[1];
            type.element_type = operands[2];
            break;
        }
        case spirv_opcode::constant:
        case spirv_opcode::spec_constant:
            OPENVINO_ASSERT(operand_count >= 3, "[GPU][Vulkan] Invalid scalar constant instruction");
            state.constants[operands[1]] = operands[2];
            break;
        case spirv_opcode::spec_constant_composite:
            OPENVINO_ASSERT(operand_count >= 3, "[GPU][Vulkan] Invalid OpSpecConstantComposite instruction");
            state.spec_constant_composites[operands[1]] = std::vector<uint32_t>(operands + 2, operands + operand_count);
            break;
        case spirv_opcode::variable:
            OPENVINO_ASSERT(operand_count >= 3, "[GPU][Vulkan] Invalid OpVariable instruction");
            state.variables.push_back({operands[0], operands[1], operands[2]});
            break;
        default:
            break;
        }
        offset += word_count;
    }

    OPENVINO_ASSERT(state.entry_point_id.has_value(), "[GPU][Vulkan] SPIR-V entry point '", requested_entry_point, "' was not found");
    return state;
}

std::optional<std::array<uint32_t, 3>> reflected_local_size_ids(const reflection_state& state) {
    std::optional<std::array<uint32_t, 3>> result;
    const auto execution_mode = state.entry_point_local_size_ids.find(*state.entry_point_id);
    if (execution_mode != state.entry_point_local_size_ids.end()) {
        result = execution_mode->second;
    }

    for (const auto& [id, decoration] : state.decorations) {
        if (decoration.built_in != static_cast<uint32_t>(spirv_builtin::workgroup_size)) {
            continue;
        }
        const auto composite = state.spec_constant_composites.find(id);
        OPENVINO_ASSERT(composite != state.spec_constant_composites.end() && composite->second.size() == 3,
                        "[GPU][Vulkan] WorkgroupSize must be a three-component specialization constant");
        std::array<uint32_t, 3> builtin_ids{};
        std::copy_n(composite->second.begin(), builtin_ids.size(), builtin_ids.begin());
        OPENVINO_ASSERT(!result.has_value() || *result == builtin_ids, "[GPU][Vulkan] SPIR-V contains conflicting local-size specialization constants");
        result = builtin_ids;
    }
    return result;
}

}  // namespace

bool vulkan_descriptor_binding::operator==(const vulkan_descriptor_binding& other) const {
    return std::tie(set, binding, type) == std::tie(other.set, other.binding, other.type);
}

bool vulkan_descriptor_binding::operator<(const vulkan_descriptor_binding& other) const {
    return std::tie(set, binding, type) < std::tie(other.set, other.binding, other.type);
}

vulkan_kernel_interface vulkan_kernel_interface::reflect(const std::vector<uint8_t>& spirv, const std::string& entry_point) {
    const auto state = parse_spirv(spirv, entry_point);
    vulkan_kernel_interface result;
    bool has_push_constant_block = false;

    for (const auto& [id, decoration] : state.decorations) {
        if (decoration.spec_id.has_value()) {
            result.specialization_ids.push_back(*decoration.spec_id);
        }
    }
    std::sort(result.specialization_ids.begin(), result.specialization_ids.end());
    result.specialization_ids.erase(std::unique(result.specialization_ids.begin(), result.specialization_ids.end()), result.specialization_ids.end());

    bool has_local_size = false;
    const auto literal_local_size = state.entry_point_local_size_literals.find(*state.entry_point_id);
    if (literal_local_size != state.entry_point_local_size_literals.end()) {
        result.local_size_defaults = literal_local_size->second;
        has_local_size = true;
    }

    if (const auto local_size_ids = reflected_local_size_ids(state)) {
        std::array<uint32_t, 3> resolved_defaults{};
        for (size_t axis = 0; axis < local_size_ids->size(); ++axis) {
            const auto constant_iterator = state.constants.find((*local_size_ids)[axis]);
            OPENVINO_ASSERT(constant_iterator != state.constants.end(), "[GPU][Vulkan] Local work-group size component is not a scalar constant");
            resolved_defaults[axis] = checked_u32(constant_iterator->second, "local work-group size");
            const auto decoration_iterator = state.decorations.find((*local_size_ids)[axis]);
            if (decoration_iterator != state.decorations.end() && decoration_iterator->second.spec_id.has_value()) {
                result.local_size_specialization_ids[axis] = *decoration_iterator->second.spec_id;
            }
        }
        OPENVINO_ASSERT(!has_local_size || result.local_size_defaults == resolved_defaults, "[GPU][Vulkan] SPIR-V contains conflicting local work-group sizes");
        result.local_size_defaults = resolved_defaults;
        has_local_size = true;
    }
    OPENVINO_ASSERT(has_local_size, "[GPU][Vulkan] Kernel '", entry_point, "' does not declare a compute local work-group size");

    for (const auto& variable : state.variables) {
        if (!state.entry_point_interface_ids.empty() && state.entry_point_interface_ids.count(variable.result_id) == 0) {
            continue;
        }
        const auto storage_class = static_cast<spirv_storage_class>(variable.storage_class);
        const auto decoration_iterator = state.decorations.find(variable.result_id);
        const auto decoration = decoration_iterator == state.decorations.end() ? decoration_set{} : decoration_iterator->second;
        if (storage_class == spirv_storage_class::storage_buffer || storage_class == spirv_storage_class::uniform) {
            if (!decoration.binding.has_value() && !decoration.descriptor_set.has_value()) {
                continue;
            }
            OPENVINO_ASSERT(decoration.binding.has_value() && decoration.descriptor_set.has_value(),
                            "[GPU][Vulkan] SPIR-V descriptor is missing a set or binding decoration");
            const auto pointer_iterator = state.types.find(variable.result_type);
            OPENVINO_ASSERT(pointer_iterator != state.types.end() && pointer_iterator->second.type_kind == type_description::kind::pointer,
                            "[GPU][Vulkan] SPIR-V descriptor variable has an invalid pointer type");
            const auto pointee_decoration_iterator = state.decorations.find(pointer_iterator->second.element_type);
            const auto is_storage_buffer = storage_class == spirv_storage_class::storage_buffer ||
                                           (pointee_decoration_iterator != state.decorations.end() && pointee_decoration_iterator->second.buffer_block);
            OPENVINO_ASSERT(is_storage_buffer, "[GPU][Vulkan] Canonical compute ABI supports storage-buffer descriptors only");
            result.descriptor_bindings.push_back({*decoration.descriptor_set, *decoration.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
        } else if (storage_class == spirv_storage_class::push_constant) {
            OPENVINO_ASSERT(!has_push_constant_block, "[GPU][Vulkan] Canonical compute ABI supports one push-constant block");
            has_push_constant_block = true;
            std::set<uint32_t> active_types;
            result.push_constant_size = reflected_type_size(variable.result_type, state, active_types);
        }
    }

    std::sort(result.descriptor_bindings.begin(), result.descriptor_bindings.end());
    result.validate_canonical_compute_abi(entry_point);
    return result;
}

void vulkan_kernel_interface::validate_canonical_compute_abi(const std::string& entry_point) const {
    OPENVINO_ASSERT(push_constant_size % sizeof(uint32_t) == 0,
                    "[GPU][Vulkan] Kernel '",
                    entry_point,
                    "' has a push-constant range that is not 32-bit aligned");
    for (size_t index = 0; index < descriptor_bindings.size(); ++index) {
        const auto& descriptor = descriptor_bindings[index];
        OPENVINO_ASSERT(descriptor.set == 0, "[GPU][Vulkan] Kernel '", entry_point, "' uses unsupported descriptor set ", descriptor.set);
        OPENVINO_ASSERT(descriptor.binding == index,
                        "[GPU][Vulkan] Kernel '",
                        entry_point,
                        "' must use consecutive storage-buffer bindings starting at zero; expected ",
                        index,
                        ", got ",
                        descriptor.binding);
        OPENVINO_ASSERT(descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        "[GPU][Vulkan] Kernel '",
                        entry_point,
                        "' uses an unsupported descriptor type at binding ",
                        descriptor.binding);
    }
}

}  // namespace cldnn::vulkan
