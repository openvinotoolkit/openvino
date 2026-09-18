// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "selective_ssm_utils.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::BufferDescriptor;
using cldnn::paged_selective_ssm;
using cldnn::primitive_impl;
using cldnn::program_node;

namespace {

constexpr size_t max_jit_state_size = 512;
// Paged recurrence amortizes wider Xe2 private state; f32 and bf16 retain SLM beyond 256 elements.
constexpr size_t max_xe2_extended_private_state_size = 256;
constexpr size_t max_xe2_f16_private_state_size = 512;
// A separate dA kernel wins only when a long recurrence amortizes its launch and temporary-buffer traffic.
constexpr size_t precompute_da_min_tokens = 3072;
constexpr size_t precompute_da_min_head_dim_groups = 12;
constexpr size_t precompute_da_min_state_size = 128;
constexpr size_t precompute_da_reference_head_dim_groups = 16;
constexpr size_t precompute_da_min_group_tokens = precompute_da_min_tokens * precompute_da_reference_head_dim_groups;

enum PagedSelectiveSSMJitStages {
    PRECOMPUTE_DA = 0,
    PRECOMPUTED_DA_RECURRENCE,
    RECURRENCE,
};

bool has_supported_indexing(const cldnn::layout& layout) {
    return layout.get_partial_shape().is_dynamic() || (layout.data_padding == cldnn::padding() && layout.count() <= std::numeric_limits<uint32_t>::max());
}

bool has_static_rank(const ov::PartialShape& shape, const size_t rank) {
    return shape.rank().is_static() && static_cast<size_t>(shape.rank().get_length()) == rank;
}

bool has_static_value(const ov::Dimension& dimension, const size_t value) {
    return dimension.is_static() && static_cast<size_t>(dimension.get_length()) == value;
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

bool use_discrete_slm(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    const size_t state_size = params.get_input_layout(2).get_partial_shape()[2].get_length();
    const auto data_type = params.get_input_layout(3).data_type;
    const bool supports_xe2_extended_private_state =
        info.arch == cldnn::gpu_arch::xe2 &&
        (state_size <= max_xe2_extended_private_state_size || (data_type == cldnn::data_types::f16 && state_size <= max_xe2_f16_private_state_size));
    const bool supports_private_state = selective_ssm_jit::supports_common_discrete_private_state(info, state_size) || supports_xe2_extended_private_state;
    return !supports_private_state;
}

size_t get_precomputed_da_head_dim_groups(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    if (info.dev_type != cldnn::device_type::discrete_gpu || info.arch < cldnn::gpu_arch::xe2)
        return 0;

    const auto& x_shape = params.get_input_layout(3).get_partial_shape();
    const auto& B_shape = params.get_input_layout(2).get_partial_shape();
    const size_t head_dim = x_shape[2].get_length();
    const size_t state_size = B_shape[2].get_length();
    if (state_size < precompute_da_min_state_size)
        return 0;

    const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(info, selective_ssm_jit::device_kind::discrete);
    const size_t head_dim_block = selective_ssm_jit::get_head_dim_block(head_dim,
                                                                        state_size,
                                                                        subgroup_size,
                                                                        info,
                                                                        selective_ssm_jit::device_kind::discrete,
                                                                        selective_ssm_jit::paged_private_value_budget);
    return head_dim_block == 0 ? 0 : cldnn::ceil_div(head_dim, head_dim_block);
}

bool supports_precomputed_da(const RuntimeParams& params) {
    return get_precomputed_da_head_dim_groups(params) >= precompute_da_min_head_dim_groups;
}

bool use_precomputed_da(const RuntimeParams& params) {
    const size_t head_dim_groups = get_precomputed_da_head_dim_groups(params);
    if (head_dim_groups < precompute_da_min_head_dim_groups)
        return false;

    if (params.is_dynamic())
        return true;

    const size_t min_tokens = std::max(precompute_da_min_tokens, cldnn::ceil_div(precompute_da_min_group_tokens, head_dim_groups));
    return static_cast<size_t>(params.get_input_layout(1).get_partial_shape()[0].get_length()) >= min_tokens;
}

class PagedSelectiveSSMJitPrecomputeGenerator : public KernelGenerator {
public:
    PagedSelectiveSSMJitPrecomputeGenerator() : KernelGenerator("paged_selective_ssm_jit_precompute") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& x_shape = params.get_input_layout(3).get_partial_shape();
        if (!params.is_dynamic())
            jit.make("SSM_TOKEN_COUNT", x_shape[0].get_length());
        jit.make("SSM_NUM_HEADS", x_shape[1].get_length());
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const size_t lws = std::min<size_t>(256, params.get_device_info().max_work_group_size);
            if (params.is_dynamic()) {
                kd.params.workGroups.global = {lws, 1, 1};
                kd.params.workGroups.local = {lws, 1, 1};
                return;
            }

            const size_t work_items = params.get_input_layout(1).count();
            kd.params.workGroups.global = {cldnn::ceil_div(work_items, lws) * lws, 1, 1};
            kd.params.workGroups.local = {lws, 1, 1};
        }};
    }
};

template <selective_ssm_jit::device_kind Kind, bool PrecomputeDA>
constexpr const char* get_paged_jit_suffix() {
    if constexpr (Kind == selective_ssm_jit::device_kind::integrated) {
        static_assert(!PrecomputeDA, "Precomputed dA is supported only by the discrete paged JIT kernel");
        return "paged_integrated";
    } else if constexpr (PrecomputeDA) {
        return "paged_discrete_precomputed_da";
    } else {
        return "paged_discrete";
    }
}

template <selective_ssm_jit::device_kind Kind, bool PrecomputeDA = false>
class PagedSelectiveSSMJitGenerator : public KernelGenerator {
public:
    PagedSelectiveSSMJitGenerator() : KernelGenerator("selective_ssm_jit", get_paged_jit_suffix<Kind, PrecomputeDA>()) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& x_shape = params.get_input_layout(3).get_partial_shape();
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        const size_t state_size = B_shape[2].get_length();
        const size_t head_dim = x_shape[2].get_length();
        const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(params.get_device_info(), Kind);
        const size_t head_dim_block = selective_ssm_jit::get_head_dim_block(head_dim,
                                                                            state_size,
                                                                            subgroup_size,
                                                                            params.get_device_info(),
                                                                            Kind,
                                                                            selective_ssm_jit::paged_private_value_budget);
        OPENVINO_ASSERT(subgroup_size != 0, "PagedSelectiveSSM JIT kernel requires a non-zero subgroup size");

        if (!params.is_dynamic())
            jit.make("SSM_TOKEN_COUNT", x_shape[0].get_length());
        jit.make("SSM_NUM_HEADS", x_shape[1].get_length());
        jit.make("SSM_HEAD_DIM", head_dim);
        jit.make("SSM_NUM_GROUPS", B_shape[1].get_length());
        jit.make("SSM_STATE_SIZE", state_size);
        jit.make("SSM_SUBGROUP_SIZE", subgroup_size);
        jit.make("SSM_HEAD_DIM_BLOCK", head_dim_block);
        // get_subgroup_size() returns 0 for unsupported devices; the clamp only keeps the divisor defined.
        jit.make("SSM_STATE_ITERATIONS", cldnn::ceil_div(state_size, std::max<size_t>(subgroup_size, 1)));
        jit.make("SSM_PAGED", true);
        jit.make("SSM_JIT_PRECOMPUTE_DA", PrecomputeDA);
        jit.make("SSM_JIT_USE_SLM", Kind == selective_ssm_jit::device_kind::discrete && use_discrete_slm(params));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        if constexpr (PrecomputeDA)
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        if constexpr (Kind == selective_ssm_jit::device_kind::discrete) {
            if (use_discrete_slm(params))
                args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const size_t num_heads = x_shape[1].get_length();
            const size_t head_dim = x_shape[2].get_length();
            const size_t state_size = B_shape[2].get_length();
            const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(params.get_device_info(), Kind);
            const size_t head_dim_block = selective_ssm_jit::get_head_dim_block(head_dim,
                                                                                state_size,
                                                                                subgroup_size,
                                                                                params.get_device_info(),
                                                                                Kind,
                                                                                selective_ssm_jit::paged_private_value_budget);
            OPENVINO_ASSERT(head_dim_block != 0, "PagedSelectiveSSM JIT kernel requires a non-zero head dimension block");

            kd.params.workGroups.local = {subgroup_size, 1, 1};
            kd.params.local_memory_args.clear();
            if constexpr (Kind == selective_ssm_jit::device_kind::discrete) {
                if (use_discrete_slm(params))
                    kd.params.local_memory_args = {head_dim_block * state_size * sizeof(float)};
            }
            if (params.is_dynamic()) {
                kd.params.workGroups.global = {subgroup_size, 1, 1};
                return;
            }

            const auto& seq_shape = params.get_input_layout(6).get_partial_shape();
            const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            // get_head_dim_block() returns 0 for unsupported configurations; the clamp only keeps the divisor defined.
            kd.params.workGroups.global = {std::max<size_t>(cldnn::ceil_div(head_dim, std::max<size_t>(head_dim_block, 1)), 1) * subgroup_size,
                                           std::max<size_t>(num_heads, 1),
                                           std::max<size_t>(sequences, 1)};
        }};
    }
};

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
                selective_ssm_utils::set_dispatch_scalars(kd, 1, params.get_device_info());
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
            const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
            if (selective_ssm_utils::requires_global_state(state_size, lws, params.get_device_info())) {
                wgs.global = {1, 1, 1};
                wgs.local = {1, 1, 1};
                selective_ssm_utils::set_dispatch_scalars(kd, 1, params.get_device_info());
                kd.params.local_memory_args = {sizeof(float)};
                return;
            }
            const size_t head_dim_block = selective_ssm_utils::get_head_dim_block(head_dim, state_size, lws, params.get_device_info());
            const size_t head_dim_groups = selective_ssm_utils::get_head_dim_groups(head_dim, head_dim_block);
            const size_t local_bytes = head_dim_block * (state_size + lws) * sizeof(float);

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(sequences, 1)};
            wgs.local = {lws, 1, 1};
            selective_ssm_utils::set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
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
                selective_ssm_utils::set_dispatch_scalars(kd, 1, params.get_device_info());
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
            const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
            const size_t head_dim_block = std::min(std::max<size_t>(head_dim, 1), selective_ssm_utils::max_head_dim_block);
            const size_t head_dim_groups = selective_ssm_utils::get_head_dim_groups(head_dim, head_dim_block);

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(sequences, 1)};
            wgs.local = {lws, 1, 1};
            selective_ssm_utils::set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
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
            const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
            if (selective_ssm_utils::requires_global_state(state_size, lws, params.get_device_info())) {
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
        const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
        return {selective_ssm_utils::requires_global_state(state_size, lws, params.get_device_info()) ? 1ul : 0ul};
    }
};

class PagedSelectiveSSMJitIntegratedImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedSelectiveSSMJitIntegratedImpl)

    Stage::Ptr paged_selective_ssm = make_stage<PagedSelectiveSSMJitGenerator<selective_ssm_jit::device_kind::integrated>>();

    PagedSelectiveSSMJitIntegratedImpl() : PrimitiveImplOCL(PagedSelectiveSSMJitIntegrated::get_type_info_static()) {}
    PagedSelectiveSSMJitIntegratedImpl(const program_node&, const RuntimeParams& params) : PagedSelectiveSSMJitIntegratedImpl() {
        add_stage(paged_selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedSelectiveSSMJitIntegratedImpl>(this);
    }
};

class PagedSelectiveSSMJitDiscreteImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedSelectiveSSMJitDiscreteImpl)

    Stage::Ptr precompute_da = make_stage<PagedSelectiveSSMJitPrecomputeGenerator>();
    Stage::Ptr paged_selective_ssm_precomputed = make_stage<PagedSelectiveSSMJitGenerator<selective_ssm_jit::device_kind::discrete, true>>();
    Stage::Ptr paged_selective_ssm = make_stage<PagedSelectiveSSMJitGenerator<selective_ssm_jit::device_kind::discrete>>();

    PagedSelectiveSSMJitDiscreteImpl() : PrimitiveImplOCL(PagedSelectiveSSMJitDiscrete::get_type_info_static()) {}
    PagedSelectiveSSMJitDiscreteImpl(const program_node&, const RuntimeParams& params) : PagedSelectiveSSMJitDiscreteImpl() {
        const bool precompute = use_precomputed_da(params);
        if (precompute) {
            add_stage(precompute_da, params);
            add_stage(paged_selective_ssm_precomputed, params);
        }
        if ((params.is_dynamic() && supports_precomputed_da(params)) || !precompute)
            add_stage(paged_selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedSelectiveSSMJitDiscreteImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        if (!has_stage(precompute_da))
            return {};

        const auto& dt_layout = params.get_input_layout(1);
        const size_t elements =
            use_precomputed_da(params) ? (dt_layout.is_dynamic() ? ov::shape_size(dt_layout.get_partial_shape().get_max_shape()) : dt_layout.count()) : 1;
        return {BufferDescriptor{elements, ov::element::f32}};
    }

    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const RuntimeParams& params) const override {
        if (use_precomputed_da(params))
            return {PRECOMPUTE_DA, PRECOMPUTED_DA_RECURRENCE};

        return {RECURRENCE};
    }
};

}  // namespace

bool validate_paged_selective_ssm_jit(const program_node& node, const selective_ssm_jit::device_kind kind) {
    const auto& info = node.get_program().get_engine().get_device_info();
    if (!selective_ssm_jit::matches_device_kind(info, kind))
        return false;

    const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(info, kind);
    if (subgroup_size == 0)
        return false;

    for (size_t i = 0; i < node.get_dependencies().size(); i++) {
        if (!has_supported_indexing(node.get_input_layout(i)))
            return false;
    }
    if (!has_supported_indexing(node.get_output_layout(0)))
        return false;

    const auto& A_shape = node.get_input_layout(0).get_partial_shape();
    const auto& dt_shape = node.get_input_layout(1).get_partial_shape();
    const auto& B_shape = node.get_input_layout(2).get_partial_shape();
    const auto& x_shape = node.get_input_layout(3).get_partial_shape();
    const auto& C_shape = node.get_input_layout(4).get_partial_shape();
    const auto& state_shape = node.get_input_layout(5).get_partial_shape();
    const auto& subsequences_shape = node.get_input_layout(6).get_partial_shape();
    const auto& block_indices_shape = node.get_input_layout(7).get_partial_shape();
    const auto& block_begins_shape = node.get_input_layout(8).get_partial_shape();
    const auto& processed_shape = node.get_input_layout(9).get_partial_shape();
    const auto& interval_shape = node.get_input_layout(10).get_partial_shape();
    const auto& output_shape = node.get_output_layout(0).get_partial_shape();
    if (!has_static_rank(A_shape, 1) || !has_static_rank(dt_shape, 2) || !has_static_rank(B_shape, 3) || !has_static_rank(x_shape, 3) ||
        !has_static_rank(C_shape, 3) || !has_static_rank(state_shape, 4) || !has_static_rank(subsequences_shape, 1) ||
        !has_static_rank(block_indices_shape, 1) || !has_static_rank(block_begins_shape, 1) || !has_static_rank(processed_shape, 1) ||
        !has_static_rank(interval_shape, 1) || !has_static_rank(output_shape, 3)) {
        return false;
    }

    if (!x_shape[1].is_static() || !x_shape[2].is_static() || !B_shape[1].is_static() || !B_shape[2].is_static())
        return false;

    const size_t num_heads = x_shape[1].get_length();
    const size_t head_dim = x_shape[2].get_length();
    const size_t num_groups = B_shape[1].get_length();
    const size_t state_size = B_shape[2].get_length();
    if (num_heads == 0 || num_heads > std::numeric_limits<uint32_t>::max() || num_groups == 0 || head_dim == 0 ||
        head_dim > std::numeric_limits<uint32_t>::max() || state_size < subgroup_size || state_size > max_jit_state_size || num_heads % num_groups != 0) {
        return false;
    }

    const bool shapes_match =
        has_static_value(A_shape[0], num_heads) && has_static_value(dt_shape[1], num_heads) && has_static_value(C_shape[1], num_groups) &&
        has_static_value(C_shape[2], state_size) && has_static_value(state_shape[1], num_heads) && has_static_value(state_shape[2], head_dim) &&
        has_static_value(state_shape[3], state_size) && has_static_value(output_shape[1], num_heads) && has_static_value(output_shape[2], head_dim) &&
        dt_shape[0].compatible(x_shape[0]) && B_shape[0].compatible(x_shape[0]) && C_shape[0].compatible(x_shape[0]) && output_shape[0].compatible(x_shape[0]);
    if (!shapes_match)
        return false;

    if (x_shape[0].is_static() && x_shape[0].get_length() == 0)
        return false;
    if (state_shape[0].is_static() && state_shape[0].get_length() == 0)
        return false;
    if (block_indices_shape[0].is_static() && block_indices_shape[0].get_length() == 0)
        return false;
    if (subsequences_shape[0].is_static()) {
        const size_t subsequences_count = subsequences_shape[0].get_length();
        if (subsequences_count < 2)
            return false;
        const size_t sequences = subsequences_count - 1;
        if ((block_begins_shape[0].is_static() && static_cast<size_t>(block_begins_shape[0].get_length()) < subsequences_count) ||
            (processed_shape[0].is_static() && static_cast<size_t>(processed_shape[0].get_length()) < sequences) ||
            (interval_shape[0].is_static() && static_cast<size_t>(interval_shape[0].get_length()) < sequences)) {
            return false;
        }
    }

    return shapes_match &&
           selective_ssm_jit::get_head_dim_block(head_dim, state_size, subgroup_size, info, kind, selective_ssm_jit::paged_private_value_budget) != 0;
}

std::unique_ptr<primitive_impl> PagedSelectiveSSMOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_selective_ssm>());
    return std::make_unique<PagedSelectiveSSMOptImpl>(node, params);
}

std::unique_ptr<primitive_impl> PagedSelectiveSSMJitIntegrated::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_selective_ssm>());
    return std::make_unique<PagedSelectiveSSMJitIntegratedImpl>(node, params);
}

std::unique_ptr<primitive_impl> PagedSelectiveSSMJitDiscrete::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_selective_ssm>());
    return std::make_unique<PagedSelectiveSSMJitDiscreteImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedSelectiveSSMOptImpl)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedSelectiveSSMJitIntegratedImpl)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedSelectiveSSMJitDiscreteImpl)
