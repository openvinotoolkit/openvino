// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "scatter_nd_update.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "kernel_selector/jitter.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "scatter_nd_update_inst.h"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

enum KernelsTypes {
    COPY_ALL = 0,
    UPDATE_REF,
    UPDATE_OPT,
};

std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = {"b", "f", "y", "x"};
    } else if (size == 5) {
        default_order = {"b", "f", "z", "y", "x"};
    } else if (size == 6) {
        default_order = {"b", "f", "w", "z", "y", "x"};
    }

    return default_order;
}

static std::string GetInputBlockND(kernel_selector::DataTensor& input, size_t shape_info_offset, size_t rank) {
    auto input_dims = input.LogicalDims();
    std::reverse(input_dims.begin(), input_dims.end());
    auto dims = input.GetDims();
    std::reverse(dims.begin(), dims.end());

    std::vector<size_t> block_nd(rank + 1);
    block_nd[rank] = 1;

    std::vector<std::string> block_nd_s(rank + 1);
    block_nd_s[rank] = "1";
    size_t input_offset = shape_info_offset;

    for (int32_t idx = static_cast<int32_t>(rank) - 1; idx >= 0; --idx) {
        block_nd[idx] = input_dims[idx] * block_nd[idx + 1];

        size_t dim_offset = idx < 2 ? idx : (kernel_selector::DataTensor::max_rank() - dims.size()) + idx;  // convert to idx in default planar format
        block_nd_s[idx] = "(" + kernel_selector::toCodeString(dims[idx], input_offset + dim_offset) + "*" + block_nd_s[idx + 1] + ")";
    }

    std::string result;
    if (input.is_dynamic()) {
        for (auto& block : block_nd_s) {
            result += block + ",";
        }
    } else {
        for (size_t block : block_nd) {
            result += kernel_selector::toCodeString(block) + ",";
        }
    }
    return result;
}

class ScatterNDUpdateCopy : public KernelGenerator {
public:
    ScatterNDUpdateCopy() : KernelGenerator("scatter_nd_update_copy") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        if (params.has_fused_primitives()) {
            FusedOpsConfiguration conf = {"", GetDefaultOrder(params.get_output_layout(0).get_rank()), "val", params.get_input_layout(0).data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_ops_arguments(args, params);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;

            const auto& in_l = params.get_input_layout(0);
            const auto& out_l = params.get_output_layout(0);
            auto b = extract_channel(ChannelName::BATCH, out_l);
            auto f = extract_channel(ChannelName::FEATURE, out_l);
            auto w = extract_channel(ChannelName::W, out_l);
            auto z = extract_channel(ChannelName::Z, out_l);
            auto y = extract_channel(ChannelName::Y, out_l);
            auto x = extract_channel(ChannelName::X, out_l);

            wgs.global = {x * y, z * w, f * b};
            std::vector<std::vector<ChannelName>> dims_by_gws = {{ChannelName::X, ChannelName::Y},
                                                                 {ChannelName::Z, ChannelName::W},
                                                                 {ChannelName::BATCH, ChannelName::FEATURE}};

            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
        }};
    }
};

static kernel_selector::MultiDataTensor get_multi_data_tensor(const RuntimeParams& params) {
    kernel_selector::MultiDataTensor inputs;
    inputs.push_back(convert_data_tensor(params.get_input_layout(0)));
    inputs.push_back(convert_data_tensor(params.get_input_layout(1)));
    inputs.push_back(convert_data_tensor(params.get_input_layout(2)));
    return inputs;
}

class ScatterNDUpdateBase : public KernelGenerator {
public:
    explicit ScatterNDUpdateBase(std::string_view name, std::string_view suffix = "") : KernelGenerator(name, suffix) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<scatter_nd_update>();
        jit.add(make_jit_constant("INDICES_RANK", desc->indices_rank));

        if (params.has_fused_primitives()) {
            FusedOpsConfiguration conf = {"", GetDefaultOrder(params.get_output_layout(0).get_rank()), "val", params.get_input_layout(0).data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        auto inputs = get_multi_data_tensor(params);
        const auto& ind_input = inputs[1];
        if (ind_input.is_dynamic()) {
            auto dims = ind_input.GetDims();
            std::reverse(dims.begin(), dims.end());

            size_t last_idx = desc->indices_rank - 1;
            size_t dim_offset = last_idx < 2 ? last_idx : last_idx + kernel_selector::DataTensor::max_rank() - desc->indices_rank;
            auto indices_last_dim =
                kernel_selector::toCodeString(dims[last_idx], dim_offset + (inputs[0].is_dynamic() ? kernel_selector::DataTensor::max_rank() : 0));
            jit.add(make_jit_constant("INDICES_LAST_DIM", indices_last_dim));
        } else {
            jit.add(make_jit_constant("INDICES_LAST_DIM", params.get_input_layout(1).get_dims()[desc->indices_rank - 1]));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        auto args = KernelGenerator::get_arguments_desc(params);
        add_fused_ops_arguments(args, params);
        return args;
    }
};

class ScatterNDUpdateRef : public ScatterNDUpdateBase {
public:
    ScatterNDUpdateRef() : ScatterNDUpdateBase("scatter_nd_update_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = ScatterNDUpdateBase::get_jit_constants(params);
        const auto& desc = params.typed_desc<scatter_nd_update>();
        auto inputs = get_multi_data_tensor(params);
        size_t input0_rank = inputs[0].LogicalDims().size();
        size_t input2_rank = inputs[2].LogicalDims().size();

        jit.add(make_jit_constant("INPUT0_BLOCK_ND", GetInputBlockND(inputs[0], inputs[0].get_dynamic_shape_offset(), input0_rank)));
        jit.add(make_jit_constant("INPUT1_BLOCK_ND", GetInputBlockND(inputs[1], inputs[1].get_dynamic_shape_offset(), desc->indices_rank - 1)));
        jit.add(make_jit_constant("INPUT2_BLOCK_ND", GetInputBlockND(inputs[2], inputs[2].get_dynamic_shape_offset(), input2_rank)));

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            const auto& desc = params.typed_desc<scatter_nd_update>();
            auto indices_dims = params.get_input_layout(1).get_dims();

            size_t indices_set_size = 1;
            for (size_t i = 0; i < (desc->indices_rank - 1); i++) {
                indices_set_size *= indices_dims[i];
            }

            auto& wgs = kd.params.workGroups;
            wgs.global = {1, 1, indices_set_size};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

class ScatterNDUpdateOpt : public ScatterNDUpdateBase {
public:
    ScatterNDUpdateOpt() : ScatterNDUpdateBase("scatter_nd_update_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        return ScatterNDUpdateBase::get_jit_constants(params);
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            const auto& updates_l = params.get_input_layout(2);
            const auto& out_l = params.get_output_layout(0);
            auto b = extract_channel(ChannelName::BATCH, updates_l);
            auto f = extract_channel(ChannelName::FEATURE, updates_l);
            auto w = extract_channel(ChannelName::W, updates_l);
            auto z = extract_channel(ChannelName::Z, updates_l);
            auto y = extract_channel(ChannelName::Y, updates_l);
            auto x = extract_channel(ChannelName::X, updates_l);

            std::vector<std::vector<ChannelName>> dims_by_gws;
            auto& wgs = kd.params.workGroups;
            wgs.global = {x * y, z * w, f * b};
            if (updates_l.get_rank() == 4) {
                wgs.global = {x, y, f * b};
                dims_by_gws = {{ChannelName::X}, {ChannelName::Y}, {ChannelName::BATCH, ChannelName::FEATURE}};
            } else if (updates_l.get_rank() == 5) {
                wgs.global = {x, y * z, f * b};
                dims_by_gws = {{ChannelName::X}, {ChannelName::Y, ChannelName::Z}, {ChannelName::BATCH, ChannelName::FEATURE}};
            } else if (updates_l.get_rank() == 6) {
                wgs.global = {x * y, z * w, f * b};
                dims_by_gws = {{ChannelName::X, ChannelName::Y}, {ChannelName::Z, ChannelName::W}, {ChannelName::BATCH, ChannelName::FEATURE}};
            } else {
                OPENVINO_THROW("Unknown rank: rank=", updates_l.get_rank());
            }

            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), updates_l.format, out_l.format, dims_by_gws);
        }};
    }
};

bool support_opt_kernel(const kernel_impl_params& params) {
    ov::Shape input_shape = params.get_input_layout(0).get_shape();
    ov::Shape indices_shape = params.get_input_layout(1).get_shape();
    ov::Shape updates_shape = params.get_input_layout(2).get_shape();
    const auto& desc = params.typed_desc<scatter_nd_update>();
    auto indices_rank = desc->indices_rank;
    const auto& last_indices_dim = indices_shape[indices_rank - 1];

    ov::Shape expected_update_shape;
    if (indices_shape.size() == 1 && indices_shape[0] == 1) {
        expected_update_shape = input_shape;
    } else {
        if (!indices_shape.empty()) {
            for (size_t i = 0; i < indices_rank - 1; ++i) {
                expected_update_shape.push_back(indices_shape[i]);
            }
        }
        if (!input_shape.empty()) {
            for (size_t i = last_indices_dim; i < input_shape.size(); ++i) {
                expected_update_shape.push_back(input_shape[i]);
            }
        }
    }

    for (size_t i = 0; i < std::min(updates_shape.size(), expected_update_shape.size()); ++i) {
        if (updates_shape[i] != expected_update_shape[i]) {
            GPU_DEBUG_TRACE << "Diff updates shape's rank::updates_shape: " << updates_shape.to_string()
                            << ", expected updates_shape: " << expected_update_shape.to_string() << std::endl;
            return false;
        }
    }
    if (ov::shape_size(updates_shape) != ov::shape_size(expected_update_shape)) {
        GPU_DEBUG_TRACE << "Diff updates shape element num::updates_shape: " << updates_shape.to_string()
                        << ", expected updates_shape: " << expected_update_shape.to_string() << std::endl;
        return false;
    }
    return true;
}

class ScatterNDUpdateImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::ScatterNDUpdateImpl)
    Stage::Ptr copy_stage = make_stage<ScatterNDUpdateCopy>();
    Stage::Ptr ref_stage = make_stage<ScatterNDUpdateRef>();
    Stage::Ptr opt_stage = make_stage<ScatterNDUpdateOpt>();

    ScatterNDUpdateImpl() : PrimitiveImplOCL(ScatterNDUpdate::get_type_info_static()) {}
    ScatterNDUpdateImpl(const program_node& node, const RuntimeParams& params) : ScatterNDUpdateImpl() {
        add_stage(copy_stage, params);
        add_stage(ref_stage, params);
        add_stage(opt_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<ScatterNDUpdateImpl>(this);
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        std::vector<size_t> stages_order = {KernelsTypes::COPY_ALL};
        auto params = instance.get_impl_params();
        stages_order.emplace_back(support_opt_kernel(*params) ? KernelsTypes::UPDATE_OPT : KernelsTypes::UPDATE_REF);
        return stages_order;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> ScatterNDUpdate::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scatter_nd_update>());
    return std::make_unique<ScatterNDUpdateImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scatter_nd_update)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::ScatterNDUpdateImpl)
