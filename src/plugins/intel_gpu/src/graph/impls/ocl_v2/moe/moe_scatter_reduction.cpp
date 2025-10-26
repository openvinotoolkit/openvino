// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_scatter_reduction.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MoeScatterReductionRefGenerator : public KernelGenerator {
public:
    MoeScatterReductionRefGenerator() : KernelGenerator("moe_scatter_reduction_ref") {}

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

    static auto calc_thread_count(RuntimeParams& params, const int vector_size, const int hidden_size) {
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

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto in_l = params.input_layouts[0];
        auto hidden_size = extract_channel(ChannelName::Y, in_l);
        auto block_size = GetBlockSize(params);
        auto [local_threads_count, batches_per_thread, unaligned_elements]  = calc_thread_count(
            const_cast<RuntimeParams&>(params), block_size, hidden_size);

        const auto& desc = params.typed_desc<moe_scatter_reduction>();

        jit.make("ACTIVE_EXPERTS", desc->num_active_experts_per_token);
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

        uint32_t num_of_inputs = 7;

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
                size_t hidden_size = params.input_layouts[0].get_shape().back();
                auto block_size = GetBlockSize(params);
                auto [local_threads_count, batches_per_thread, unaligned_elements]  = calc_thread_count(
                    const_cast<RuntimeParams&>(params), block_size, hidden_size);

                auto num_tokens = extract_channel(ChannelName::BATCH, params.input_layouts[1]);

                wgs.global = {num_tokens * local_threads_count, 1, 1};
                wgs.local = { local_threads_count, 1, 1};
            }
        }};
    }
};

class MoeScatterReductionRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeScatterReductionRefImpl)

    Stage::Ptr moe_scatter_reduction = make_stage<MoeScatterReductionRefGenerator>();

    MoeScatterReductionRefImpl() : PrimitiveImplOCL(MoeScatterReductionRef::get_type_info_static()) {}
    MoeScatterReductionRefImpl(const program_node& node, const RuntimeParams& params) : MoeScatterReductionRefImpl() {
        add_stage(moe_scatter_reduction, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeScatterReductionRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeScatterReductionRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_scatter_reduction>());
    return std::make_unique<MoeScatterReductionRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_scatter_reduction)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeScatterReductionRefImpl)
