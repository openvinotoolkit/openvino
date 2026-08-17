// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t max_head_dim_block = 4;

size_t get_lws(const size_t state_size, const device_info& info) {
    const size_t limit = std::min<size_t>(32, info.max_work_group_size);
    const size_t target = std::min(std::max<size_t>(state_size, 1), limit);
    size_t lws = 1;
    while (lws * 2 <= target)
        lws *= 2;
    return lws;
}

bool requires_global_state(const size_t state_size, const size_t lws, const device_info& info) {
    const size_t local_capacity = info.max_local_mem_size / sizeof(float);
    return state_size > std::numeric_limits<uint32_t>::max() || state_size > local_capacity || lws > local_capacity - state_size;
}

size_t get_scratch_elements(const std::array<size_t, 4>& dimensions) {
    size_t elements = 1;
    for (const size_t dimension : dimensions) {
        OPENVINO_ASSERT(dimension == 0 || elements <= std::numeric_limits<size_t>::max() / dimension,
                        "PagedSelectiveSSM global state scratch size overflows size_t");
        elements *= dimension;
    }
    return std::max<size_t>(elements, 1);
}

size_t get_head_dim_block(const size_t head_dim, const size_t state_size, const size_t lws, const device_info& info) {
    const size_t local_capacity = info.max_local_mem_size / sizeof(float);
    const size_t state_and_reduction = state_size + lws;
    size_t block = std::min(std::max<size_t>(head_dim, 1), max_head_dim_block);
    while (block > 1 && state_and_reduction > local_capacity / block)
        --block;
    return block;
}

void set_dispatch_scalars(KernelData& kd, const size_t block, const device_info& info) {
    kd.params.scalars.clear();
    scalar_desc block_desc;
    block_desc.t = scalar_desc::Types::INT32;
    block_desc.v.s32 = static_cast<int32_t>(block);
    kd.params.scalars.push_back(block_desc);

    const bool use_subgroup_reduction = info.dev_type == device_type::integrated_gpu || info.gfx_ver.major >= 20;
    scalar_desc reduction_desc;
    reduction_desc.t = scalar_desc::Types::UINT32;
    reduction_desc.v.u32 = use_subgroup_reduction ? 1 : 0;
    kd.params.scalars.push_back(reduction_desc);
}

class PagedSelectiveSSMOptGenerator : public KernelGenerator {
public:
    PagedSelectiveSSMOptGenerator() : KernelGenerator("paged_selective_ssm_opt") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            auto& wgs = kd.params.workGroups;
            if (params.is_dynamic()) {
                wgs.global = {1, 1, 1};
                wgs.local = {1, 1, 1};
                set_dispatch_scalars(kd, 1, params.get_device_info());
                kd.params.local_memory_args = {2 * sizeof(float)};
                return;
            }

            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(6).get_partial_shape();
            const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            const size_t num_heads = x_shape[1].get_length();
            const size_t head_dim = x_shape[2].get_length();
            const size_t state_size = B_shape[2].get_length();
            const size_t lws = get_lws(state_size, params.get_device_info());
            if (requires_global_state(state_size, lws, params.get_device_info())) {
                wgs.global = {1, 1, 1};
                wgs.local = {1, 1, 1};
                set_dispatch_scalars(kd, 1, params.get_device_info());
                kd.params.local_memory_args = {sizeof(float)};
                return;
            }
            const size_t head_dim_block = get_head_dim_block(head_dim, state_size, lws, params.get_device_info());
            const size_t head_dim_groups = head_dim / head_dim_block + static_cast<size_t>(head_dim % head_dim_block != 0);
            const size_t local_bytes = head_dim_block * (state_size + lws) * sizeof(float);

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(sequences, 1)};
            wgs.local = {lws, 1, 1};
            set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
            kd.params.local_memory_args = {local_bytes};
        }};
    }
};

class PagedSelectiveSSMLargeStateGenerator : public KernelGenerator {
public:
    PagedSelectiveSSMLargeStateGenerator() : KernelGenerator("paged_selective_ssm_large_state") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            auto& wgs = kd.params.workGroups;
            if (params.is_dynamic()) {
                wgs.global = {1, 1, 1};
                wgs.local = {1, 1, 1};
                set_dispatch_scalars(kd, 1, params.get_device_info());
                kd.params.local_memory_args = {sizeof(float)};
                return;
            }

            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(6).get_partial_shape();
            const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            const size_t num_heads = x_shape[1].get_length();
            const size_t head_dim = x_shape[2].get_length();
            const size_t state_size = B_shape[2].get_length();
            const size_t lws = get_lws(state_size, params.get_device_info());
            const size_t head_dim_block = std::min(std::max<size_t>(head_dim, 1), max_head_dim_block);
            const size_t head_dim_groups = head_dim / head_dim_block + static_cast<size_t>(head_dim % head_dim_block != 0);

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(sequences, 1)};
            wgs.local = {lws, 1, 1};
            set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
            kd.params.local_memory_args = {head_dim_block * lws * sizeof(float)};
        }};
    }
};

class PagedSelectiveSSMOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedSelectiveSSMOptImpl)

    Stage::Ptr paged_selective_ssm = make_stage<PagedSelectiveSSMOptGenerator>();
    Stage::Ptr paged_selective_ssm_large_state = make_stage<PagedSelectiveSSMLargeStateGenerator>();

    PagedSelectiveSSMOptImpl() : PrimitiveImplOCL(PagedSelectiveSSMOpt::get_type_info_static()) {}
    PagedSelectiveSSMOptImpl(const program_node&, const RuntimeParams& params) : PagedSelectiveSSMOptImpl() {
        add_stage(paged_selective_ssm, params);
        add_stage(paged_selective_ssm_large_state, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedSelectiveSSMOptImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        size_t state_scratch_elements = 1;
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        const auto& x_shape = params.get_input_layout(3).get_partial_shape();
        const auto& seq_shape = params.get_input_layout(6).get_partial_shape();
        if (B_shape.is_static() && x_shape.is_static() && seq_shape.is_static()) {
            const size_t state_size = B_shape[2].get_length();
            const size_t lws = get_lws(state_size, params.get_device_info());
            if (requires_global_state(state_size, lws, params.get_device_info())) {
                const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
                const size_t num_heads = x_shape[1].get_length();
                const size_t head_dim = x_shape[2].get_length();
                state_scratch_elements = get_scratch_elements({sequences, num_heads, head_dim, state_size});
            }
        }
        return {BufferDescriptor{state_scratch_elements, ov::element::f32}};
    }

    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const RuntimeParams& params) const override {
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        if (!B_shape.is_static())
            return {0};
        const size_t state_size = B_shape[2].get_length();
        const size_t lws = get_lws(state_size, params.get_device_info());
        return {requires_global_state(state_size, lws, params.get_device_info()) ? 1ul : 0ul};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PagedSelectiveSSMOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_selective_ssm>());
    return std::make_unique<PagedSelectiveSSMOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedSelectiveSSMOptImpl)
