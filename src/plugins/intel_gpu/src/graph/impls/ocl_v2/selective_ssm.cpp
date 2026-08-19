// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>

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

}  // namespace

std::unique_ptr<primitive_impl> SelectiveSSMOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMOptImpl)
