// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "moe_mask_gen.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MoeMaskGenRefGenerator : public KernelGenerator {
public:
    MoeMaskGenRefGenerator() : KernelGenerator("moe_mask_gen") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        auto prim = params.typed_desc<moe_mask_gen>();
        jit.make("NUM_EXPERTS_PER_TOKEN", prim->num_experts_per_token);
        jit.make("ONEDNN_GROUPED_GEMM_USED", prim->onednn_grouped_gemm_used);
        jit.make("EXPERT_SORT", prim->expert_sort);
        jit.make("ROW_LUT_ENABLE", prim->expert_sort);

        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});

        const auto num_of_outputs = static_cast<uint32_t>(params.typed_desc<moe_mask_gen>()->output_size());
        for (uint32_t i = 0; i < num_of_outputs; i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                auto num_total_experts = static_cast<size_t>(params.typed_desc<moe_mask_gen>()->num_total_experts);
                wgs.global = {num_total_experts, 1, 1};
                wgs.local = {num_total_experts, 1, 1};
            }
        }};
    }
};

// Multi work-group expert sort: one work-group per expert_sort_block_size rows
constexpr size_t expert_sort_block_size = 256;
// Up to this many rows the single work-group kernel is used: one launch instead of two (decode)
constexpr size_t expert_sort_single_wg_max_rows = 64;

size_t get_expert_sort_rows(const RuntimeParams& params) {
    const auto& shape = params.get_input_layout(0).get_shape();
    return shape[0] * static_cast<size_t>(params.typed_desc<moe_mask_gen>()->num_experts_per_token);
}

size_t get_expert_sort_blocks(const RuntimeParams& params) {
    return std::max<size_t>(1, ceil_div(get_expert_sort_rows(params), expert_sort_block_size));
}

class MoeExpertSortGenerator : public KernelGenerator {
public:
    explicit MoeExpertSortGenerator(bool scatter) : KernelGenerator("moe_mask_gen", scatter ? "expert_sort_scatter" : "expert_sort_hist"), m_scatter(scatter) {}

protected:
    bool m_scatter;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto prim = params.typed_desc<moe_mask_gen>();
        jit.make(m_scatter ? "EXPERT_SORT_SCATTER" : "EXPERT_SORT_HIST", 1);
        jit.make("NUM_EXPERTS_PER_TOKEN", prim->num_experts_per_token);
        jit.make("NUM_EXPERTS", prim->num_total_experts);
        jit.make("BLOCK_SIZE", expert_sort_block_size);
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        if (m_scatter) {
            for (uint32_t i = 0; i < 3; i++) {
                args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            }
        }
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            if (!params.is_dynamic()) {
                wgs.global = {get_expert_sort_blocks(params) * expert_sort_block_size, 1, 1};
                wgs.local = {expert_sort_block_size, 1, 1};
            }
        }};
    }
};

class MoeMaskGenRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeMaskGenRefImpl)

    enum Stages : size_t { SINGLE_WG = 0, EXPERT_SORT_HIST = 1, EXPERT_SORT_SCATTER = 2 };
    Stage::Ptr single_wg = make_stage<MoeMaskGenRefGenerator>();
    Stage::Ptr expert_sort_hist = make_stage<MoeExpertSortGenerator>(false);
    Stage::Ptr expert_sort_scatter = make_stage<MoeExpertSortGenerator>(true);

    MoeMaskGenRefImpl() : PrimitiveImplOCL(MoeMaskGenRef::get_type_info_static()) {}
    MoeMaskGenRefImpl(const program_node& node, const RuntimeParams& params) : MoeMaskGenRefImpl() {
        add_stage(single_wg, params);
        if (params.typed_desc<moe_mask_gen>()->expert_sort) {
            add_stage(expert_sort_hist, params);
            add_stage(expert_sort_scatter, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeMaskGenRefImpl>(this);
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::kernel_impl_params& params) const override {
        if (!params.typed_desc<moe_mask_gen>()->expert_sort || params.is_dynamic() || get_expert_sort_rows(params) <= expert_sort_single_wg_max_rows) {
            return {SINGLE_WG};
        }
        return {EXPERT_SORT_HIST, EXPERT_SORT_SCATTER};
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto prim = params.typed_desc<moe_mask_gen>();
        if (!prim->expert_sort) {
            return {};
        }
        const size_t blocks = params.is_dynamic() ? 1 : get_expert_sort_blocks(params);
        return {BufferDescriptor{blocks * static_cast<size_t>(prim->num_total_experts), ov::element::i32}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeMaskGenRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_mask_gen>());
    return std::make_unique<MoeMaskGenRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_mask_gen)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeMaskGenRefImpl)
