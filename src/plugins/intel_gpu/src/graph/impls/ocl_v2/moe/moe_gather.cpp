// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_gather.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_gather.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MoeGatherRefGenerator : public KernelGenerator {
public:
    MoeGatherRefGenerator() : KernelGenerator("moe_gather_ref") {}

protected:
    static size_t GetBlockSize(const RuntimeParams& params) {
        const auto& input = params.get_input_layout(0);
        size_t vec_size = 1;
        switch (input.data_type) {
        case ov::element::i8:
        case ov::element::u8:
            vec_size = 16;
            break;
        case ov::element::f16:
            vec_size = 8;
            break;
        case ov::element::f32:
        case ov::element::i32:
            vec_size = 4;
            break;
        case ov::element::i64:
            vec_size = 2;
            break;
        default:
            vec_size = 1;
            break;
        }
        return vec_size;
    }

    static auto calc_thread_count(RuntimeParams& params, const size_t vector_size, const size_t hidden_size) {
        auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
        const uint64_t threads_needed = (hidden_size + vector_size - 1) / vector_size;
        size_t local_threads_needed = std::min(threads_needed, max_wgs);
        size_t batches_per_thread = 1;
        size_t unaligned_elements = 0;

        if (threads_needed <= max_wgs) {
            batches_per_thread = 1;
            unaligned_elements = hidden_size % vector_size;
        } else {
            batches_per_thread = (threads_needed + max_wgs - 1) / max_wgs;
            auto new_block_size = batches_per_thread * vector_size;
            unaligned_elements = hidden_size % new_block_size;

            local_threads_needed = hidden_size / new_block_size;
            auto partialblock = (hidden_size % new_block_size != 0) ? 1 : 0;
            local_threads_needed += partialblock;
        }

        return std::tuple{local_threads_needed, batches_per_thread, unaligned_elements};
    }

    static size_t get_hidden_size(const RuntimeParams& params) {
        const auto& input_layout = params.get_input_layout(0);
        size_t input_rank = input_layout.get_partial_shape().size();
        return input_layout.get_partial_shape()[input_rank - 1].get_length();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto in_l = params.input_layouts[0];
        auto hidden_size = get_hidden_size(params);
        auto block_size = GetBlockSize(params);
        auto [local_threads_count, batches_per_thread, unaligned_elements] = calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);

        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", block_size);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);
        jit.make("UNALIGNED_ELEMENTS", unaligned_elements);

        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        const uint32_t num_of_inputs = 2;
        for (uint32_t i = 0; i < num_of_inputs; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                auto hidden_size = get_hidden_size(params);
                auto block_size = GetBlockSize(params);
                auto [local_threads_count, batches_per_thread, unaligned_elements] =
                    calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);
                auto token_per_expert = extract_channel(ChannelName::BATCH, params.input_layouts[1]);

                wgs.global = {token_per_expert * local_threads_count, 1, 1};
                wgs.local = {local_threads_count, 1, 1};
            }
        }};
    }
};

class MoeGatherRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeGatherRefImpl)

    Stage::Ptr moe_gather = make_stage<MoeGatherRefGenerator>();

    MoeGatherRefImpl() : PrimitiveImplOCL(MoeGatherRef::get_type_info_static()) {}
    MoeGatherRefImpl(const program_node& node, const RuntimeParams& params) : MoeGatherRefImpl() {
        add_stage(moe_gather, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeGatherRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeGatherRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_gather>());
    return std::make_unique<MoeGatherRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gather)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeGatherRefImpl)
