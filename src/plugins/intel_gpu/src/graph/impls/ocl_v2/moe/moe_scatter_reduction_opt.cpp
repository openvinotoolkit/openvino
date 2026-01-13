// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_scatter_reduction_opt.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"

namespace ov::intel_gpu::ocl {
namespace {

using namespace ov::intel_gpu::ocl;
class MoeScatterReductionOptGenerator : public KernelGenerator {
public:
    MoeScatterReductionOptGenerator() : KernelGenerator("moe_scatter_reduction_opt") {}

protected:
    static auto calc_thread_count(RuntimeParams& params, const size_t vector_size, const size_t hidden_size) {
        auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
        const uint64_t threads_needed = (hidden_size + vector_size - 1) / vector_size;
        size_t local_threads_needed = std::min(threads_needed, max_wgs);
        size_t batches_per_thread = 1;

        if (threads_needed <= max_wgs) {
            batches_per_thread = 1;
        } else {
            batches_per_thread = (threads_needed + max_wgs - 1) / max_wgs;
            auto new_block_size = batches_per_thread * vector_size;

            local_threads_needed = hidden_size / new_block_size;
            auto partialblock = (hidden_size % new_block_size != 0) ? 1 : 0;
            local_threads_needed += static_cast<size_t>(partialblock);
        }

        return std::tuple{local_threads_needed, batches_per_thread};
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto in_l = params.input_layouts[0];
        auto hidden_size = extract_channel(ChannelName::Y, in_l);
        auto [local_threads_count, batches_per_thread] = calc_thread_count(const_cast<RuntimeParams&>(params), MoeScatterReductionOpt::block_size, hidden_size);

        const auto& desc = params.typed_desc<moe_scatter_reduction>();

        jit.make("ACTIVE_EXPERTS", desc->num_active_experts_per_token);
        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", MoeScatterReductionOpt::block_size);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);

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
                auto [local_threads_count, batches_per_thread] =
                    calc_thread_count(const_cast<RuntimeParams&>(params), MoeScatterReductionOpt::block_size, hidden_size);

                auto num_tokens = extract_channel(ChannelName::BATCH, params.input_layouts[1]);

                wgs.global = {num_tokens * local_threads_count, 1, 1};
                wgs.local = {local_threads_count, 1, 1};
            }
        }};
    }
};

class MoeScatterReductionOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeScatterReductionOptImpl)

    Stage::Ptr moe_scatter_reduction = make_stage<MoeScatterReductionOptGenerator>();

    MoeScatterReductionOptImpl() : PrimitiveImplOCL(MoeScatterReductionOpt::get_type_info_static()) {}
    MoeScatterReductionOptImpl(const program_node& node, const RuntimeParams& params) : MoeScatterReductionOptImpl() {
        add_stage(moe_scatter_reduction, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeScatterReductionOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeScatterReductionOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_scatter_reduction>());
    return std::make_unique<MoeScatterReductionOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeScatterReductionOptImpl)
