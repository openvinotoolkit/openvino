// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_spirv_reflection.hpp"

#include <algorithm>
#include <unordered_map>

namespace cldnn {
namespace vk {

namespace {

constexpr uint32_t kSpvMagic = 0x07230203;

enum SpvOp : uint16_t {
    OpString = 7,
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpTypeInt = 21,
    OpConstant = 43,
};

// NonSemantic.ClspvReflection.5 extended instruction opcodes (revision 7).
enum ClspvExtInst : uint32_t {
    ExtInstKernel = 1,
    ExtInstArgumentInfo = 2,
    ExtInstArgumentStorageBuffer = 3,
    ExtInstArgumentUniform = 4,
    ExtInstArgumentPodStorageBuffer = 5,
    ExtInstArgumentPodUniform = 6,
    ExtInstArgumentPodPushConstant = 7,
    ExtInstArgumentSampledImage = 8,
    ExtInstArgumentStorageImage = 9,
    ExtInstArgumentSampler = 10,
    ExtInstArgumentWorkgroup = 11,
    ExtInstSpecConstantWorkgroupSize = 12,
    ExtInstSpecConstantGlobalOffset = 13,
    ExtInstSpecConstantWorkDim = 14,
    ExtInstPushConstantGlobalOffset = 15,
    ExtInstPushConstantEnqueuedLocalSize = 16,
    ExtInstPushConstantGlobalSize = 17,
    ExtInstPushConstantRegionOffset = 18,
    ExtInstPushConstantNumWorkgroups = 19,
    ExtInstPushConstantRegionGroupOffset = 20,
    ExtInstConstantDataStorageBuffer = 21,
    ExtInstConstantDataUniform = 22,
    ExtInstLiteralSampler = 23,
    ExtInstPropertyRequiredWorkgroupSize = 24,
    ExtInstSpecConstantSubgroupMaxSize = 25,
    ExtInstArgumentPointerPushConstant = 26,
    ExtInstArgumentPointerUniform = 27,
    ExtInstProgramScopeVariablesStorageBuffer = 28,
    ExtInstProgramScopeVariablePointerRelocation = 29,
    ExtInstImageArgumentInfoChannelOrderPushConstant = 30,
    ExtInstImageArgumentInfoChannelDataTypePushConstant = 31,
    ExtInstImageArgumentInfoChannelOrderUniform = 32,
    ExtInstImageArgumentInfoChannelDataTypeUniform = 33,
    ExtInstArgumentStorageTexelBuffer = 34,
    ExtInstArgumentUniformTexelBuffer = 35,
    ExtInstConstantDataPointerPushConstant = 36,
    ExtInstProgramScopeVariablePointerPushConstant = 37,
    ExtInstPrintfInfo = 38,
    ExtInstPrintfBufferStorageBuffer = 39,
    ExtInstPrintfBufferPointerPushConstant = 40,
    ExtInstNormalizedSamplerMaskPushConstant = 41,
    ExtInstWorkgroupVariableSize = 42,
};

class reflection_parser {
public:
    explicit reflection_parser(const std::vector<uint32_t>& words) : words_(words) {}

    std::vector<vk_kernel_reflection> parse() {
        if (words_.size() < 5 || words_[0] != kSpvMagic)
            return {};
        size_t idx = 5;
        while (idx < words_.size()) {
            uint32_t header = words_[idx];
            uint32_t word_count = header >> 16;
            uint16_t opcode = header & 0xFFFF;
            if (word_count == 0 || idx + word_count > words_.size())
                return {};  // malformed
            parse_instruction(opcode, words_.data() + idx, word_count);
            idx += word_count;
        }
        return finish();
    }

private:
    const std::vector<uint32_t>& words_;

    // id -> u32 literal for OpConstant of 32-bit unsigned type.
    std::unordered_map<uint32_t, uint32_t> constants_;
    // id -> string for OpString.
    std::unordered_map<uint32_t, std::string> strings_;
    // kernel function id -> reflection.
    std::unordered_map<uint32_t, size_t> kernels_;
    // result id of the Kernel extended instruction (decl id) -> function id.
    std::unordered_map<uint32_t, uint32_t> decl_to_func_;
    // id of the ClspvReflection extended instruction set import.
    uint32_t reflection_import_ = 0;
    std::vector<vk_kernel_reflection> result_;
    uint32_t uint32_type_id_ = 0;

    void parse_instruction(uint16_t opcode, const uint32_t* w, uint32_t word_count) {
        switch (opcode) {
        case OpString: {
            if (word_count < 3)
                return;
            uint32_t id = w[1];
            const char* str = reinterpret_cast<const char*>(w + 2);
            strings_[id] = std::string(str);
            break;
        }
        case OpExtInstImport: {
            if (word_count < 3)
                return;
            const char* name = reinterpret_cast<const char*>(w + 2);
            if (name == std::string("NonSemantic.ClspvReflection.5"))
                reflection_import_ = w[1];
            break;
        }
        case OpExtInst: {
            if (word_count < 6)
                return;
            // w[1]=type, w[2]=result id, w[3]=set id, w[4]=ext opcode, w[5..]=operands
            if (w[3] != reflection_import_)
                break;
            parse_ext_inst(w[4], w[2], w + 5, word_count - 5);
            break;
        }
        case OpEntryPoint: {
            // w[1]=execution model, w[2]=entry point id, w[3..]=name
            if (word_count < 4)
                return;
            uint32_t id = w[2];
            const char* name = reinterpret_cast<const char*>(w + 3);
            auto& refl = result_.emplace_back();
            refl.name = std::string(name);
            kernels_[id] = result_.size() - 1;
            break;
        }
        case OpExecutionMode: {
            // w[1]=entry point id, w[2]=mode, w[3..]=mode operands
            if (word_count < 4)
                return;
            uint32_t mode = w[2];
            if (mode == 17) {  // LocalSize
                auto it = kernels_.find(w[1]);
                if (it != kernels_.end() && word_count >= 7) {
                    auto& refl = result_[it->second];
                    refl.local_size[0] = w[3];
                    refl.local_size[1] = w[4];
                    refl.local_size[2] = w[5];
                    refl.has_local_size = true;
                }
            }
            break;
        }
        case OpTypeInt: {
            if (word_count >= 4 && w[2] == 32 && w[3] == 0)
                uint32_type_id_ = w[1];
            break;
        }
        case OpConstant: {
            // w[1]=type, w[2]=result id, w[3]=value
            if (word_count >= 4 && w[1] == uint32_type_id_)
                constants_[w[2]] = w[3];
            break;
        }
        default:
            break;
        }
    }

    vk_arg_kind kind_for(uint32_t ext_inst) const {
        switch (ext_inst) {
        case ExtInstArgumentStorageBuffer: return vk_arg_kind::storage_buffer;
        case ExtInstArgumentUniform: return vk_arg_kind::uniform_buffer;
        case ExtInstArgumentPodStorageBuffer: return vk_arg_kind::pod_storage_buffer;
        case ExtInstArgumentPodUniform: return vk_arg_kind::pod_uniform;
        case ExtInstArgumentPodPushConstant: return vk_arg_kind::pod_push_constant;
        case ExtInstArgumentSampledImage: return vk_arg_kind::sampled_image;
        case ExtInstArgumentStorageImage: return vk_arg_kind::storage_image;
        case ExtInstArgumentSampler: return vk_arg_kind::sampler;
        case ExtInstArgumentWorkgroup: return vk_arg_kind::workgroup;
        case ExtInstArgumentPointerPushConstant: return vk_arg_kind::pointer_push_constant;
        case ExtInstArgumentPointerUniform: return vk_arg_kind::pointer_uniform;
        case ExtInstArgumentStorageTexelBuffer: return vk_arg_kind::storage_texel_buffer;
        case ExtInstArgumentUniformTexelBuffer: return vk_arg_kind::uniform_texel_buffer;
        default: return vk_arg_kind::storage_buffer;
        }
    }

    vk_kernel_reflection* find_kernel(uint32_t id) {
        auto k = kernels_.find(id);
        if (k != kernels_.end())
            return &result_[k->second];
        auto it = decl_to_func_.find(id);
        if (it != decl_to_func_.end()) {
            k = kernels_.find(it->second);
            if (k != kernels_.end())
                return &result_[k->second];
        }
        return nullptr;
    }

    void add_arg(uint32_t kernel_id, vk_kernel_arg arg) {
        auto* refl = find_kernel(kernel_id);
        if (refl == nullptr)
            return;
        refl->args.push_back(std::move(arg));
    }

    void parse_ext_inst(uint32_t ext_inst, uint32_t result_id, const uint32_t* op, uint32_t count) {
        auto get = [this](uint32_t id) { return constants_.count(id) ? constants_[id] : 0u; };
        // Argument declarations reference the result id of the Kernel extended
        // instruction (decl id), not the entry point function id.
        switch (ext_inst) {
        case ExtInstKernel: {
            // op[0]=function id, op[1]=name string id
            if (count >= 2) {
                decl_to_func_[result_id] = op[0];
                auto it = kernels_.find(op[0]);
                if (it != kernels_.end() && strings_.count(op[1]))
                    result_[it->second].name = strings_[op[1]];
            }
            break;
        }
        case ExtInstPropertyRequiredWorkgroupSize: {
            // op[0]=kernel decl id, op[1..3]=x,y,z
            if (count >= 4) {
                auto* refl = find_kernel(op[0]);
                if (refl != nullptr) {
                    refl->local_size[0] = get(op[1]);
                    refl->local_size[1] = get(op[2]);
                    refl->local_size[2] = get(op[3]);
                    refl->has_local_size = true;
                }
            }
            break;
        }
        case ExtInstSpecConstantWorkgroupSize: {
            // op[0..2]=spec ids of workgroup size x,y,z (module-wide, no kernel arg).
            // clspv drops literal OpExecutionMode LocalSize for the whole module
            // when any kernel reads gl_WorkGroupSize from spec constants, so every
            // kernel must be specialized with its required size at pipeline build.
            if (count >= 3) {
                for (auto& refl : result_)
                    refl.uses_spec_wgsize = true;
            }
            break;
        }
        case ExtInstArgumentStorageBuffer:
        case ExtInstArgumentUniform:
        case ExtInstArgumentSampledImage:
        case ExtInstArgumentStorageImage:
        case ExtInstArgumentSampler:
        case ExtInstArgumentStorageTexelBuffer:
        case ExtInstArgumentUniformTexelBuffer: {
            // op[0]=kernel, op[1]=ordinal, op[2]=descriptor set, op[3]=binding, op[4]?=arginfo
            if (count >= 4) {
                vk_kernel_arg arg;
                arg.ordinal = get(op[1]);
                arg.kind = kind_for(ext_inst);
                arg.descriptor_set = get(op[2]);
                arg.binding = get(op[3]);
                if (count >= 5 && strings_.count(op[4]))
                    arg.name = strings_[op[4]];
                add_arg(op[0], std::move(arg));
            }
            break;
        }
        case ExtInstArgumentPodStorageBuffer:
        case ExtInstArgumentPodUniform: {
            // op[0]=kernel, op[1]=ordinal, op[2]=ds, op[3]=binding, op[4]=offset, op[5]=size, op[6]?=arginfo
            if (count >= 6) {
                vk_kernel_arg arg;
                arg.ordinal = get(op[1]);
                arg.kind = kind_for(ext_inst);
                arg.descriptor_set = get(op[2]);
                arg.binding = get(op[3]);
                arg.offset = get(op[4]);
                arg.size = get(op[5]);
                if (count >= 7 && strings_.count(op[6]))
                    arg.name = strings_[op[6]];
                add_arg(op[0], std::move(arg));
            }
            break;
        }
        case ExtInstArgumentPodPushConstant:
        case ExtInstArgumentPointerPushConstant: {
            // op[0]=kernel, op[1]=ordinal, op[2]=offset, op[3]=size, op[4]?=arginfo
            if (count >= 4) {
                vk_kernel_arg arg;
                arg.ordinal = get(op[1]);
                arg.kind = kind_for(ext_inst);
                arg.offset = get(op[2]);
                arg.size = get(op[3]);
                if (count >= 5 && strings_.count(op[4]))
                    arg.name = strings_[op[4]];
                add_arg(op[0], std::move(arg));
            }
            break;
        }
        case ExtInstArgumentPointerUniform: {
            // op[0]=kernel, op[1]=ordinal, op[2]=ds, op[3]=binding, op[4]=offset, op[5]=size, op[6]?=arginfo
            if (count >= 6) {
                vk_kernel_arg arg;
                arg.ordinal = get(op[1]);
                arg.kind = kind_for(ext_inst);
                arg.descriptor_set = get(op[2]);
                arg.binding = get(op[3]);
                arg.offset = get(op[4]);
                arg.size = get(op[5]);
                if (count >= 7 && strings_.count(op[6]))
                    arg.name = strings_[op[6]];
                add_arg(op[0], std::move(arg));
            }
            break;
        }
        case ExtInstArgumentWorkgroup: {
            // op[0]=kernel, op[1]=ordinal, op[2]=spec id, op[3]=elem size, op[4]?=arginfo
            if (count >= 4) {
                vk_kernel_arg arg;
                arg.ordinal = get(op[1]);
                arg.kind = vk_arg_kind::workgroup;
                arg.offset = get(op[2]);  // spec id
                arg.size = get(op[3]);    // element size
                if (count >= 5 && strings_.count(op[4]))
                    arg.name = strings_[op[4]];
                add_arg(op[0], std::move(arg));
            }
            break;
        }
        default:
            break;  // ignore spec constants, push constant descriptions, constant data, printf
        }
    }

    std::vector<vk_kernel_reflection> finish() {
        for (auto& refl : result_) {
            std::sort(refl.args.begin(), refl.args.end(),
                      [](const vk_kernel_arg& a, const vk_kernel_arg& b) { return a.ordinal < b.ordinal; });
        }
        return std::move(result_);
    }
};

}  // namespace

const char* arg_kind_name(vk_arg_kind kind) {
    switch (kind) {
    case vk_arg_kind::storage_buffer: return "storage_buffer";
    case vk_arg_kind::uniform_buffer: return "uniform_buffer";
    case vk_arg_kind::pod_storage_buffer: return "pod_storage_buffer";
    case vk_arg_kind::pod_uniform: return "pod_uniform";
    case vk_arg_kind::pod_push_constant: return "pod_push_constant";
    case vk_arg_kind::sampled_image: return "sampled_image";
    case vk_arg_kind::storage_image: return "storage_image";
    case vk_arg_kind::sampler: return "sampler";
    case vk_arg_kind::workgroup: return "workgroup";
    case vk_arg_kind::pointer_push_constant: return "pointer_push_constant";
    case vk_arg_kind::pointer_uniform: return "pointer_uniform";
    case vk_arg_kind::storage_texel_buffer: return "storage_texel_buffer";
    case vk_arg_kind::uniform_texel_buffer: return "uniform_texel_buffer";
    }
    return "unknown";
}

uint32_t vk_kernel_reflection::max_binding() const {
    uint32_t max = 0;
    for (const auto& arg : args) {
        switch (arg.kind) {
        case vk_arg_kind::storage_buffer:
        case vk_arg_kind::uniform_buffer:
        case vk_arg_kind::pod_storage_buffer:
        case vk_arg_kind::pod_uniform:
        case vk_arg_kind::sampled_image:
        case vk_arg_kind::storage_image:
        case vk_arg_kind::sampler:
        case vk_arg_kind::storage_texel_buffer:
        case vk_arg_kind::uniform_texel_buffer:
            max = std::max(max, arg.binding + 1);
            break;
        default:
            break;
        }
    }
    return max;
}

uint32_t vk_kernel_reflection::push_constants_size() const {
    uint32_t max = 0;
    for (const auto& arg : args) {
        if (arg.kind == vk_arg_kind::pod_push_constant || arg.kind == vk_arg_kind::pointer_push_constant)
            max = std::max(max, arg.offset + arg.size);
    }
    return max;
}

std::vector<vk_kernel_reflection> parse_spirv_reflection(const std::vector<uint32_t>& spirv) {
    reflection_parser parser(spirv);
    return parser.parse();
}

}  // namespace vk
}  // namespace cldnn
