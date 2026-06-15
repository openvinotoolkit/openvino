// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_router_fused_opt.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class MoeRouterSoftMaxTopK : public KernelGenerator {
public:
    MoeRouterSoftMaxTopK() : KernelGenerator("moe_router_fused", "softmax_topk") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_router_fused>();
        jit.make("SOFTMAX_TOPK_ENABLE", 1);
        jit.make("TOP_K", desc->_config.top_k);
        jit.make("VALUE_NUM", desc->_config.num_expert);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoeRouterSigmoidBiasTopK : public KernelGenerator {
public:
    MoeRouterSigmoidBiasTopK() : KernelGenerator("moe_router_fused", "sigmoid_bias_topk") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_router_fused>();
        jit.make("SIGMOID_BIAS_TOPK_ENABLE", 1);
        jit.make("TOP_K", desc->_config.top_k);
        jit.make("VALUE_NUM", desc->_config.num_expert);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoeRouterFusedImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeRouterFusedImpl)

    Stage::Ptr softmax_topk = make_stage<MoeRouterSoftMaxTopK>();
    Stage::Ptr sigmoid_bias_topk = make_stage<MoeRouterSigmoidBiasTopK>();

    MoeRouterFusedImpl() : PrimitiveImplOCL(MoeRouterFusedOpt::get_type_info_static()) {}
    MoeRouterFusedImpl(const program_node& node, const RuntimeParams& params) : MoeRouterFusedImpl() {
        auto desc = params.typed_desc<moe_router_fused>();
        if (desc->_config.routing_type == MoERouterFused::RoutingType::SOFTMAX) {
            add_stage(softmax_topk, params);
        } else {
            OPENVINO_ASSERT(desc->_config.routing_type == MoERouterFused::RoutingType::SIGMOID_BIAS, "Unsupported routing type");
            add_stage(sigmoid_bias_topk, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoeRouterFusedImpl>(this);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        kernel_dump_info.clear_entries();

        cldnn::stream& stream = instance.get_network().get_stream();
        auto desc = instance.get_typed_desc<moe_router_fused>();
        const auto& config = desc->_config;

        // Determine token count from input layout
        auto input_mem = instance.input_memory_ptr(0);
        auto input_layout = instance.dependencies()[0].first->get_impl_params()->get_output_layout(instance.dependencies()[0].second);
        auto input_shape = input_layout.get_shape();
        size_t token_num = input_shape[0];
        if (input_shape.size() >= 3)
            token_num = input_shape[0] * input_shape[1];
        size_t lws_size = config.num_expert;

        // Select routing stage and build inputs
        Stage* routing_stage = nullptr;
        std::vector<memory::ptr> inputs;
        if (config.routing_type == MoERouterFused::RoutingType::SOFTMAX) {
            routing_stage = softmax_topk.get();
            inputs = {input_mem};
        } else {
            routing_stage = sigmoid_bias_topk.get();
            inputs = {input_mem, instance.input_memory_ptr(1), instance.input_memory_ptr(2)};
        }

        // Kernel signature: (input, output_index, output_weights)
        // MoERouterFused outputs: 0 = topk_weights, 1 = topk_indices
        auto topk_weights_mem = instance.output_memory_ptr(0);
        auto topk_indices_mem = instance.output_memory_ptr(1);

        // Build kernel arguments manually (same as execute_stage in moe_3gemm)
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc kargs_desc;

        for (uint32_t i = 0; i < inputs.size(); i++) {
            kargs_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }
        // Kernel outputs: first = indices (u32), second = weights (f16/f32)
        kargs_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.outputs.push_back(topk_indices_mem);
        kargs_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        args.outputs.push_back(topk_weights_mem);

        stream.set_arguments(*routing_stage->kernel, kargs_desc, args);
        kargs_desc.workGroups.global = {token_num, lws_size};
        kargs_desc.workGroups.local = {1, lws_size};

        kernel_dump_info.add_entry_point(routing_stage->kernel->get_id());

        return stream.enqueue_kernel(*routing_stage->kernel, kargs_desc, {}, events, instance.needs_completion_event());
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeRouterFusedOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_router_fused>());
    return std::make_unique<MoeRouterFusedImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl
