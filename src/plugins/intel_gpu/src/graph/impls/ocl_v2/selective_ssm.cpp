// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>
#include <limits>

#include "intel_gpu/primitives/selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "selective_ssm_utils.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::BufferDescriptor;
using cldnn::primitive_impl;
using cldnn::program_node;
using cldnn::selective_ssm;

namespace {

constexpr size_t max_jit_state_size = 512;
// Larger Xe2 private state is profitable once recurrence work amortizes its scratch traffic.
constexpr size_t max_xe2_extended_private_state_size = 256;
constexpr size_t xe2_extended_private_min_sequence = 16;
constexpr size_t xe2_extended_bf16_private_min_sequence = 8;
constexpr size_t xe2_short_sequence_head_dim_block = 2;
constexpr size_t xe2_short_sequence_limit = 1;

bool is_plain_static_layout(const cldnn::layout& layout) {
    return layout.get_partial_shape().is_static() && layout.data_padding == cldnn::padding() && layout.count() <= std::numeric_limits<uint32_t>::max();
}

bool has_static_rank(const ov::PartialShape& shape, const size_t rank) {
    return shape.is_static() && shape.rank().is_static() && static_cast<size_t>(shape.rank().get_length()) == rank;
}

size_t get_private_values_budget(const size_t sequence_size, const cldnn::data_types data_type, const cldnn::device_info& info) {
    const bool supports_extended_f32_short_sequence = info.gfx_ver.major > 12 || (info.gfx_ver.major == 12 && info.gfx_ver.minor >= 70);
    const size_t short_sequence_limit = data_type == cldnn::data_types::f32 && supports_extended_f32_short_sequence ? 32 : 8;
    return sequence_size <= short_sequence_limit ? selective_ssm_jit::short_sequence_private_value_budget
                                                 : selective_ssm_jit::long_sequence_private_value_budget;
}

bool use_discrete_slm(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    const size_t state_size = params.get_input_layout(2).get_partial_shape()[3].get_length();
    const size_t sequence_size = params.get_input_layout(3).get_partial_shape()[1].get_length();
    const auto data_type = params.get_input_layout(3).data_type;
    const size_t extended_private_min_sequence =
        data_type == cldnn::data_types::bf16 ? xe2_extended_bf16_private_min_sequence : xe2_extended_private_min_sequence;
    const bool supports_xe2_extended_private_state =
        info.arch == cldnn::gpu_arch::xe2 && state_size <= max_xe2_extended_private_state_size && sequence_size >= extended_private_min_sequence;
    // Xe HPG does not amortize the f32 private-state scratch cost for very short recurrences.
    const bool prefer_slm_for_short_f32 = info.arch == cldnn::gpu_arch::xe_hpg && data_type == cldnn::data_types::f32 && sequence_size <= 4;
    const bool supports_private_state = selective_ssm_jit::supports_common_discrete_private_state(info, state_size) || supports_xe2_extended_private_state;
    return !supports_private_state || prefer_slm_for_short_f32;
}

size_t get_discrete_head_dim_target(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    if (info.dev_type != cldnn::device_type::discrete_gpu)
        return selective_ssm_jit::default_discrete_head_dim_block;

    const size_t sequence_size = params.get_input_layout(3).get_partial_shape()[1].get_length();
    const bool use_short_sequence_block = info.arch == cldnn::gpu_arch::xe2 && !use_discrete_slm(params) && sequence_size <= xe2_short_sequence_limit;
    return use_short_sequence_block ? xe2_short_sequence_head_dim_block : selective_ssm_jit::default_discrete_head_dim_block;
}

template <selective_ssm_jit::device_kind Kind>
class SelectiveSSMJitGenerator : public KernelGenerator {
public:
    SelectiveSSMJitGenerator() : KernelGenerator("selective_ssm_jit", Kind == selective_ssm_jit::device_kind::integrated ? "integrated" : "discrete") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& x_shape = params.get_input_layout(3).get_partial_shape();
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        const size_t state_size = B_shape[3].get_length();
        const size_t head_dim = x_shape[3].get_length();
        const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(params.get_device_info(), Kind);
        const size_t private_values_budget = get_private_values_budget(x_shape[1].get_length(), params.get_input_layout(3).data_type, params.get_device_info());
        const size_t discrete_target = get_discrete_head_dim_target(params);
        const size_t head_dim_block =
            selective_ssm_jit::get_head_dim_block(head_dim, state_size, subgroup_size, params.get_device_info(), Kind, private_values_budget, discrete_target);
        OPENVINO_ASSERT(subgroup_size != 0, "SelectiveSSM JIT kernel requires a non-zero subgroup size");

        jit.make("SSM_SEQUENCE_SIZE", x_shape[1].get_length());
        jit.make("SSM_NUM_HEADS", x_shape[2].get_length());
        jit.make("SSM_HEAD_DIM", head_dim);
        jit.make("SSM_NUM_GROUPS", B_shape[2].get_length());
        jit.make("SSM_STATE_SIZE", state_size);
        jit.make("SSM_SUBGROUP_SIZE", subgroup_size);
        jit.make("SSM_HEAD_DIM_BLOCK", head_dim_block);
        // get_subgroup_size() returns 0 for unsupported devices; the clamp only keeps the divisor defined.
        jit.make("SSM_STATE_ITERATIONS", cldnn::ceil_div(state_size, std::max<size_t>(subgroup_size, 1)));
        jit.make("SSM_PAGED", false);
        jit.make("SSM_JIT_PRECOMPUTE_DA", false);
        jit.make("SSM_JIT_USE_SLM", Kind == selective_ssm_jit::device_kind::discrete && use_discrete_slm(params));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        OPENVINO_ASSERT(!params.is_dynamic(), "SelectiveSSM JIT kernel requires static shapes");
        Arguments args;
        for (uint32_t i = 0; i < params.input_layouts.size(); i++)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        for (uint32_t i = 0; i < params.output_layouts.size(); i++)
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        if constexpr (Kind == selective_ssm_jit::device_kind::discrete) {
            if (use_discrete_slm(params))
                args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            OPENVINO_ASSERT(!params.is_dynamic(), "SelectiveSSM JIT kernel requires static shapes");
            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const size_t batch = x_shape[0].get_length();
            const size_t num_heads = x_shape[2].get_length();
            const size_t head_dim = x_shape[3].get_length();
            const size_t state_size = B_shape[3].get_length();
            const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(params.get_device_info(), Kind);
            const size_t private_values_budget =
                get_private_values_budget(x_shape[1].get_length(), params.get_input_layout(3).data_type, params.get_device_info());
            const size_t discrete_target = get_discrete_head_dim_target(params);
            const size_t head_dim_block = selective_ssm_jit::get_head_dim_block(head_dim,
                                                                                state_size,
                                                                                subgroup_size,
                                                                                params.get_device_info(),
                                                                                Kind,
                                                                                private_values_budget,
                                                                                discrete_target);
            OPENVINO_ASSERT(head_dim_block != 0, "SelectiveSSM JIT kernel requires a non-zero head dimension block");

            // get_head_dim_block() returns 0 for unsupported configurations; the clamp only keeps the divisor defined.
            kd.params.workGroups.global = {cldnn::ceil_div(head_dim, std::max<size_t>(head_dim_block, 1)) * subgroup_size, num_heads, batch};
            kd.params.workGroups.local = {subgroup_size, 1, 1};
            kd.params.scalars.clear();
            cldnn::scalar_desc sequence_desc;
            sequence_desc.t = cldnn::scalar_desc::Types::UINT32;
            sequence_desc.v.u32 = static_cast<uint32_t>(x_shape[1].get_length());
            kd.params.scalars.push_back(sequence_desc);
            kd.params.local_memory_args.clear();
            if constexpr (Kind == selective_ssm_jit::device_kind::discrete) {
                if (use_discrete_slm(params))
                    kd.params.local_memory_args = {head_dim_block * state_size * sizeof(float)};
            }
        }};
    }
};

class SelectiveSSMOptGenerator : public KernelGenerator {
public:
    SelectiveSSMOptGenerator() : KernelGenerator("selective_ssm_opt") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
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
            const size_t batch = x_shape[0].get_length();
            const size_t num_heads = x_shape[2].get_length();
            const size_t head_dim = x_shape[3].get_length();
            const size_t state_size = B_shape[3].get_length();
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

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(batch, 1)};
            wgs.local = {lws, 1, 1};
            selective_ssm_utils::set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
            kd.params.local_memory_args = {local_bytes};
        }};
    }
};

class SelectiveSSMLargeStateGenerator : public KernelGenerator {
public:
    SelectiveSSMLargeStateGenerator() : KernelGenerator("selective_ssm_large_state") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
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
            const size_t batch = x_shape[0].get_length();
            const size_t num_heads = x_shape[2].get_length();
            const size_t head_dim = x_shape[3].get_length();
            const size_t state_size = B_shape[3].get_length();
            const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
            const size_t head_dim_block = std::min(std::max<size_t>(head_dim, 1), selective_ssm_utils::max_head_dim_block);
            const size_t head_dim_groups = selective_ssm_utils::get_head_dim_groups(head_dim, head_dim_block);

            wgs.global = {std::max<size_t>(head_dim_groups, 1) * lws, std::max<size_t>(num_heads, 1), std::max<size_t>(batch, 1)};
            wgs.local = {lws, 1, 1};
            selective_ssm_utils::set_dispatch_scalars(kd, head_dim_block, params.get_device_info());
            kd.params.local_memory_args = {head_dim_block * lws * sizeof(float)};
        }};
    }
};

class SelectiveSSMOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SelectiveSSMOptImpl)

    Stage::Ptr selective_ssm = make_stage<SelectiveSSMOptGenerator>();
    Stage::Ptr selective_ssm_large_state = make_stage<SelectiveSSMLargeStateGenerator>();

    SelectiveSSMOptImpl() : PrimitiveImplOCL(SelectiveSSMOpt::get_type_info_static()) {}
    SelectiveSSMOptImpl(const program_node&, const RuntimeParams& params) : SelectiveSSMOptImpl() {
        add_stage(selective_ssm, params);
        add_stage(selective_ssm_large_state, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SelectiveSSMOptImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        size_t state_scratch_elements = 1;
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        const auto& state_layout = params.get_input_layout(5);
        if (B_shape.is_static() && state_layout.is_static()) {
            const size_t state_size = B_shape[3].get_length();
            const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
            if (selective_ssm_utils::requires_global_state(state_size, lws, params.get_device_info()))
                state_scratch_elements = std::max<size_t>(state_layout.count(), 1);
        }

        return {BufferDescriptor{state_scratch_elements, ov::element::f32}};
    }

    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const RuntimeParams& params) const override {
        const auto& B_shape = params.get_input_layout(2).get_partial_shape();
        if (!B_shape.is_static())
            return {0};
        const size_t state_size = B_shape[3].get_length();
        const size_t lws = selective_ssm_utils::get_lws(state_size, params.get_device_info());
        return {selective_ssm_utils::requires_global_state(state_size, lws, params.get_device_info()) ? 1ul : 0ul};
    }
};

class SelectiveSSMJitIntegratedImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SelectiveSSMJitIntegratedImpl)

    Stage::Ptr selective_ssm = make_stage<SelectiveSSMJitGenerator<selective_ssm_jit::device_kind::integrated>>();

    SelectiveSSMJitIntegratedImpl() : PrimitiveImplOCL(SelectiveSSMJitIntegrated::get_type_info_static()) {}
    SelectiveSSMJitIntegratedImpl(const program_node&, const RuntimeParams& params) : SelectiveSSMJitIntegratedImpl() {
        add_stage(selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SelectiveSSMJitIntegratedImpl>(this);
    }
};

class SelectiveSSMJitDiscreteImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SelectiveSSMJitDiscreteImpl)

    Stage::Ptr selective_ssm = make_stage<SelectiveSSMJitGenerator<selective_ssm_jit::device_kind::discrete>>();

    SelectiveSSMJitDiscreteImpl() : PrimitiveImplOCL(SelectiveSSMJitDiscrete::get_type_info_static()) {}
    SelectiveSSMJitDiscreteImpl(const program_node&, const RuntimeParams& params) : SelectiveSSMJitDiscreteImpl() {
        add_stage(selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SelectiveSSMJitDiscreteImpl>(this);
    }
};

}  // namespace

bool validate_selective_ssm_jit(const program_node& node, const selective_ssm_jit::device_kind kind) {
    const auto& info = node.get_program().get_engine().get_device_info();
    if (!selective_ssm_jit::matches_device_kind(info, kind))
        return false;

    const size_t subgroup_size = selective_ssm_jit::get_subgroup_size(info, kind);
    if (subgroup_size == 0)
        return false;

    for (size_t i = 0; i < node.get_dependencies().size(); i++) {
        if (!is_plain_static_layout(node.get_input_layout(i)))
            return false;
    }
    for (size_t i = 0; i < node.get_outputs_count(); i++) {
        if (!is_plain_static_layout(node.get_output_layout(i)))
            return false;
    }

    const auto& A_shape = node.get_input_layout(0).get_partial_shape();
    const auto& dt_shape = node.get_input_layout(1).get_partial_shape();
    const auto& B_shape = node.get_input_layout(2).get_partial_shape();
    const auto& x_shape = node.get_input_layout(3).get_partial_shape();
    const auto& C_shape = node.get_input_layout(4).get_partial_shape();
    const auto& state_shape = node.get_input_layout(5).get_partial_shape();
    const auto& output_shape = node.get_output_layout(0).get_partial_shape();
    const auto& output_state_shape = node.get_output_layout(1).get_partial_shape();
    if (!has_static_rank(A_shape, 1) || !has_static_rank(dt_shape, 3) || !has_static_rank(B_shape, 4) || !has_static_rank(x_shape, 4) ||
        !has_static_rank(C_shape, 4) || !has_static_rank(state_shape, 4) || !has_static_rank(output_shape, 4) || !has_static_rank(output_state_shape, 4)) {
        return false;
    }

    const size_t batch = x_shape[0].get_length();
    const size_t sequence = x_shape[1].get_length();
    const size_t num_heads = x_shape[2].get_length();
    const size_t head_dim = x_shape[3].get_length();
    const size_t num_groups = B_shape[2].get_length();
    const size_t state_size = B_shape[3].get_length();
    if (batch == 0 || sequence == 0 || num_heads == 0 || num_groups == 0 || head_dim == 0 || state_size < subgroup_size || state_size > max_jit_state_size ||
        num_heads % num_groups != 0) {
        return false;
    }

    const bool shapes_match = A_shape[0] == num_heads && dt_shape[0] == batch && dt_shape[1] == sequence && dt_shape[2] == num_heads && B_shape[0] == batch &&
                              B_shape[1] == sequence && C_shape == B_shape && state_shape[0] == batch && state_shape[1] == num_heads &&
                              state_shape[2] == head_dim && state_shape[3] == state_size && output_shape == x_shape && output_state_shape == state_shape;
    const size_t private_values_budget = get_private_values_budget(sequence, node.get_input_layout(3).data_type, info);
    return shapes_match && selective_ssm_jit::get_head_dim_block(head_dim, state_size, subgroup_size, info, kind, private_values_budget) != 0;
}

std::unique_ptr<primitive_impl> SelectiveSSMOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMOptImpl>(node, params);
}

std::unique_ptr<primitive_impl> SelectiveSSMJitIntegrated::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMJitIntegratedImpl>(node, params);
}

std::unique_ptr<primitive_impl> SelectiveSSMJitDiscrete::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMJitDiscreteImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMOptImpl)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMJitIntegratedImpl)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMJitDiscreteImpl)
