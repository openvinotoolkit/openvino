// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_scatter_reduction_row_lut.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

std::pair<size_t, size_t> get_dispatch(const RuntimeParams& params) {
    const auto hidden_size = static_cast<size_t>(params.input_layouts[0].get_partial_shape()[1].get_length());
    const auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
    return MoeScatterReductionRowLut::get_dispatch(max_wgs, hidden_size);
}

class MoeScatterReductionRowLutGenerator : public KernelGenerator {
public:
    MoeScatterReductionRowLutGenerator() : KernelGenerator("moe_scatter_reduction_row_lut") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.make("ACTIVE_EXPERTS", params.typed_desc<moe_scatter_reduction>()->num_active_experts_per_token);
        jit.make("HIDDEN_SIZE", params.input_layouts[0].get_partial_shape()[1].get_length());
        jit.make("VEC_BLK_SIZE", MoeScatterReductionRowLut::block_size);
        jit.make("BATCHES_PER_THREAD", get_dispatch(params).second);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        for (uint32_t i = 0; i < 3; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                const auto local_threads = get_dispatch(params).first;
                const auto num_tokens = params.output_layouts[0].get_shape()[0];
                wgs.global = {num_tokens * local_threads, 1, 1};
                wgs.local = {local_threads, 1, 1};
            }
        }};
    }
};

class MoeScatterReductionRowLutImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeScatterReductionRowLutImpl)

    Stage::Ptr scatter_reduce = make_stage<MoeScatterReductionRowLutGenerator>();

    MoeScatterReductionRowLutImpl() : PrimitiveImplOCL(MoeScatterReductionRowLut::get_type_info_static()) {}
    MoeScatterReductionRowLutImpl(const program_node& node, const RuntimeParams& params) : MoeScatterReductionRowLutImpl() {
        add_stage(scatter_reduce, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeScatterReductionRowLutImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeScatterReductionRowLut::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_scatter_reduction>());
    return std::make_unique<MoeScatterReductionRowLutImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeScatterReductionRowLutImpl)
