// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_router_fused_opt.hpp"

#include <algorithm>
#include "../common_utils/dispatch_utils.hpp"
#include "../common_utils/jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include <cstdlib>
#include <chrono>
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "moe_otd_runtime.hpp"
#include "openvino/op/constant.hpp"

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

class MoeLesProxyFC : public KernelGenerator {
public:
    MoeLesProxyFC() : KernelGenerator("moe_les_proxy_fc") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.make("LES_PROXY_FC_ENABLE", 1);
        jit.make("LES_PROXY_EXPERTS", params.typed_desc<moe_router_fused>()->_config.num_expert);
        jit.make("LES_PROXY_SUBGROUP_SIZE", 16);
        jit.make("LES_PROXY_GROUP_SIZE", 64);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }

};

class MoeRouterFusedImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoeRouterFusedImpl)

    Stage::Ptr softmax_topk = make_stage<MoeRouterSoftMaxTopK>();
    Stage::Ptr sigmoid_bias_topk = make_stage<MoeRouterSigmoidBiasTopK>();
    Stage::Ptr les_proxy_fc;
    program::les_router_link les_proxy_link;
    memory::ptr les_proxy_weights;
    memory::ptr les_proxy_weights_source;
    memory::ptr les_proxy_logits;
    memory::ptr les_proxy_indices;
    memory::ptr les_proxy_routing_weights;

    memory::ptr get_proxy_weights(cldnn::network& network, const program::les_router_link& link) {
        auto weights = network.get_primitive(link.weights)->output_memory_ptr();
        auto projection_instance = network.get_primitive(link.projection);
        auto projection = projection_instance->get_typed_desc<fully_connected>();
        if (!projection->compressed_weights) {
            return weights;
        }
        if (les_proxy_weights && les_proxy_weights_source == weights) {
            return les_proxy_weights;
        }
        auto& stream = network.get_stream();
        const auto read_values = [&stream](const memory::ptr& parameter) {
            const auto& parameter_layout = parameter->get_layout();
            std::vector<uint8_t> bytes(parameter_layout.bytes_count());
            parameter->copy_to(stream, bytes.data(), 0, 0, bytes.size(), true);
            ov::op::v0::Constant constant(parameter_layout.data_type, parameter_layout.get_shape(), bytes.data());
            return constant.cast_vector<float>();
        };
        const auto& dependencies = projection_instance->dependencies();
        const size_t scale_index = projection->bias.is_valid() ? 3U : 2U;
        auto scale_memory = dependencies[scale_index].first->output_memory_ptr(dependencies[scale_index].second);
        auto packed = read_values(weights);
        auto scales = read_values(scale_memory);
        auto zero_memory = projection->decompression_zero_point.is_valid()
                               ? dependencies[scale_index + 1].first->output_memory_ptr(dependencies[scale_index + 1].second)
                               : memory::ptr{};
        auto zeros = zero_memory ? read_values(zero_memory)
                                 : std::vector<float>{projection->decompression_zero_point_scalar.value_or(0.0f)};
        const size_t hidden_size = weights->get_layout().get_shape()[1];
        const auto parameter_index = [&](const memory::ptr& parameter, size_t count, size_t expert, size_t hidden) {
            if (count == 1) {
                return size_t{0};
            }
            const size_t groups = count / link.experts;
            const size_t group = hidden / (hidden_size / groups);
            const auto& parameter_layout = parameter->get_layout();
            const bool experts_first = parameter_layout.get_shape().front() == link.experts;
            cldnn::tensor coordinates(0);
            coordinates.batch[0] = static_cast<int32_t>(experts_first ? expert : group);
            coordinates.feature[0] = static_cast<int32_t>(experts_first ? group : expert);
            return parameter_layout.get_linear_offset(coordinates);
        };
        std::vector<ov::float16> decompressed(packed.size());
        for (size_t expert = 0; expert < link.experts; ++expert) {
            for (size_t hidden = 0; hidden < hidden_size; ++hidden) {
                const size_t index = expert * hidden_size + hidden;
                const auto difference = ov::float16(packed[index] - zeros[parameter_index(zero_memory, zeros.size(), expert, hidden)]);
                decompressed[index] = ov::float16(static_cast<float>(difference) *
                    scales[parameter_index(scale_memory, scales.size(), expert, hidden)]);
            }
        }
        les_proxy_weights = network.get_engine().allocate_memory(
            cldnn::layout{weights->get_layout().get_partial_shape(), data_types::f16, format::bfyx},
            network.get_engine().get_preferred_memory_allocation_type(), false);
        les_proxy_weights->copy_from(stream, decompressed.data(), true);
        les_proxy_weights_source = weights;
        if (auto* perf = moe_otd::get_perf_counters()) {
            perf->les_proxy_weight_initializations.fetch_add(1, std::memory_order_relaxed);
        }
        return les_proxy_weights;
    }

    MoeRouterFusedImpl() : PrimitiveImplOCL(MoeRouterFusedOpt::get_type_info_static()) {
        les_proxy_fc = make_stage<MoeLesProxyFC>();
    }
    MoeRouterFusedImpl(const program_node& node, const RuntimeParams& params) : MoeRouterFusedImpl() {
        auto desc = params.typed_desc<moe_router_fused>();
        const auto* prediction = std::getenv("OV_GPU_LES_PREDICT");
        if (params.prog->get_config().get_enable_les() && prediction && std::strcmp(prediction, "1") == 0) {
            add_stage(les_proxy_fc, params);
            const auto router_links = params.prog->get_les_router_links();
            for (const auto& link : router_links) {
                if (link.router == node.id() && link.status == program::les_router_status::supported) {
                    les_proxy_link = link;
                    break;
                }
            }
        }
        GPU_DEBUG_INFO << "[LES_PROXY_IMPL] router=" << node.id() << " enabled="
                   << params.prog->get_config().get_enable_les() << " stage=" << static_cast<bool>(les_proxy_fc)
                       << " status=" << static_cast<int>(les_proxy_link.status) << std::endl;
        if (desc->_config.routing_type == MoERouterFused::RoutingType::SOFTMAX) {
            add_stage(softmax_topk, params);
        } else {
            OPENVINO_ASSERT(desc->_config.routing_type == MoERouterFused::RoutingType::SIGMOID_BIAS, "Unsupported routing type");
            add_stage(sigmoid_bias_topk, params);
        }
        GPU_DEBUG_INFO << "[LES_PROXY_IMPL] router=" << node.id() << " activated_stages=" << _order.size()
                       << " total_stages=" << _stages.size() << std::endl;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<MoeRouterFusedImpl>(this);
        copy->les_proxy_logits.reset();
        copy->les_proxy_indices.reset();
        copy->les_proxy_routing_weights.reset();
        return copy;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        return execute_routing(events, instance, false);
    }

    void enqueue_les_prediction(cldnn::primitive_inst& instance) override {
        execute_routing({}, instance, true);
    }

    cldnn::event::ptr execute_routing(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance, bool proxy_only) {
        kernel_dump_info.clear_entries();

        auto& network = instance.get_network();
        cldnn::stream& stream = network.get_stream();
        auto desc = instance.get_typed_desc<moe_router_fused>();
        const auto& config = desc->_config;

        // Determine token count from input layout
        auto input_mem = proxy_only ? memory::ptr{} : instance.input_memory_ptr(0);
        size_t token_num = 1;
        if (!proxy_only) {
            auto input_layout = instance.dependencies()[0].first->get_impl_params()->get_output_layout(instance.dependencies()[0].second);
            auto input_shape = input_layout.get_shape();
            token_num = input_shape[0];
            if (input_shape.size() >= 3) {
                token_num = input_shape[0] * input_shape[1];
            }
        }
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
        auto topk_weights_mem = proxy_only ? memory::ptr{} : instance.output_memory_ptr(0);
        auto topk_indices_mem = proxy_only ? memory::ptr{} : instance.output_memory_ptr(1);
        cldnn::event::ptr proxy_prediction_ready;
        const program::les_router_link* active_proxy_link = nullptr;

        const auto runtime_link = std::find_if(network.get_les_router_links().begin(),
                                               network.get_les_router_links().end(),
                                               [&](const auto& link) {
                                                   return link.router == instance.id() &&
                                                          link.status == program::les_router_status::supported;
                                               });
        if (runtime_link != network.get_les_router_links().end()) {
            proxy_prediction_ready = network.get_les_proxy_state(runtime_link->chain.target_moe).prediction_ready;
        }
        bool prediction_created = false;
        if (!proxy_prediction_ready && token_num == 1 && network.is_les_prediction_enabled() && network.get_les_context() &&
            runtime_link != network.get_les_router_links().end()) {
            OPENVINO_ASSERT(les_proxy_fc, "Qualified LES proxy Router stage was not compiled for ", instance.id());
            const auto& proxy_link = *runtime_link;
            active_proxy_link = &proxy_link;
            auto state = network.get_les_proxy_state(proxy_link.chain.target_moe);
            if (state.hidden && state.hidden_ready &&
                state.hidden->get_layout().count() == state.hidden->get_layout().get_shape().back()) {
                const bool profile_proxy = moe_otd::les_trace_enabled() && network.get_config().get_enable_profiling();
                const auto submit_start = profile_proxy ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                auto weights = get_proxy_weights(network, proxy_link);
                auto bias = proxy_link.projection_bias.empty()
                                ? memory::ptr{}
                                : network.get_primitive(proxy_link.projection_bias)->output_memory_ptr();
                const auto logits_layout = cldnn::layout({1, 1, 1, static_cast<int64_t>(proxy_link.experts)},
                                                          state.hidden->get_layout().data_type,
                                                          cldnn::format::bfyx);
                const auto reuse = [&network](memory::ptr& memory, const cldnn::layout& desired, bool lockable = false) {
                    const auto allocation = lockable ? network.get_engine().get_lockable_preferred_memory_allocation_type()
                                                     : network.get_engine().get_preferred_memory_allocation_type();
                    if (!memory || memory.use_count() > 1 || memory->get_layout() != desired ||
                        memory->get_allocation_type() != allocation) {
                        memory = network.get_engine().allocate_memory(
                            desired, allocation, false);
                    }
                    return memory;
                };
                auto logits = reuse(les_proxy_logits, logits_layout);
                auto proxy_indices = reuse(les_proxy_indices, cldnn::layout(
                    {1, 1, 1, static_cast<int64_t>(proxy_link.top_k)},
                    instance.get_impl_params()->get_output_layout(1).data_type, cldnn::format::bfyx),
                    network.is_les_prefetch_enabled());
                auto proxy_weights = reuse(les_proxy_routing_weights, cldnn::layout(
                    {1, 1, 1, static_cast<int64_t>(proxy_link.top_k)},
                    state.hidden->get_layout().data_type, cldnn::format::bfyx));

                cldnn::kernel_arguments_data fc_args;
                cldnn::kernel_arguments_desc fc_desc;
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
                fc_args.inputs.push_back(state.hidden);
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
                fc_args.inputs.push_back(weights);
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
                fc_args.inputs.push_back(bias ? bias : weights);
                cldnn::scalars_desc fc_scalars(2);
                fc_scalars[0].t = ScalarDescriptor::Types::UINT32;
                fc_scalars[0].v.u32 = static_cast<uint32_t>(state.hidden->get_layout().get_shape().back());
                fc_scalars[1].t = ScalarDescriptor::Types::UINT32;
                fc_scalars[1].v.u32 = bias ? 1U : 0U;
                fc_args.scalars = &fc_scalars;
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
                fc_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
                fc_args.outputs.push_back(logits);
                stream.set_arguments(*les_proxy_fc->kernel, fc_desc, fc_args);
                fc_desc.workGroups.global = {proxy_link.experts * 64, 1, 1};
                fc_desc.workGroups.local = {64, 1, 1};
                auto proxy_projection = stream.enqueue_kernel(*les_proxy_fc->kernel,
                                                              fc_desc,
                                                              fc_args,
                                                              {state.hidden_ready},
                                                              true);

                std::vector<memory::ptr> proxy_inputs{logits};
                if (config.routing_type == MoERouterFused::RoutingType::SIGMOID_BIAS) {
                    proxy_inputs.push_back(instance.input_memory_ptr(1));
                    proxy_inputs.push_back(instance.input_memory_ptr(2));
                }
                cldnn::kernel_arguments_data proxy_args;
                cldnn::kernel_arguments_desc proxy_desc;
                for (uint32_t i = 0; i < proxy_inputs.size(); ++i) {
                    proxy_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
                    proxy_args.inputs.push_back(proxy_inputs[i]);
                }
                proxy_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
                proxy_args.outputs.push_back(proxy_indices);
                proxy_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
                proxy_args.outputs.push_back(proxy_weights);
                stream.set_arguments(*routing_stage->kernel, proxy_desc, proxy_args);
                proxy_desc.workGroups.global = {1, lws_size};
                proxy_desc.workGroups.local = {1, lws_size};
                auto prediction_ready = stream.enqueue_kernel(*routing_stage->kernel,
                                                             proxy_desc,
                                                             proxy_args,
                                                             {proxy_projection},
                                                             true);
                if (profile_proxy) {
                    const auto submit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - submit_start).count();
                    const auto execution_ns = [](const cldnn::event::ptr& kernel_event) {
                        kernel_event->wait();
                        uint64_t duration = 0;
                        for (const auto& interval : kernel_event->get_profiling_info()) {
                            if (interval.stage == cldnn::instrumentation::profiling_stage::executing) {
                                duration += interval.value->value().count();
                            }
                        }
                        return duration;
                    };
                    const auto fc_ns = execution_ns(proxy_projection);
                    const auto topk_ns = execution_ns(prediction_ready);
                    if (auto* perf = moe_otd::get_perf_counters()) {
                        perf->les_proxy_fc_ns.fetch_add(fc_ns, std::memory_order_relaxed);
                        perf->les_proxy_topk_ns.fetch_add(topk_ns, std::memory_order_relaxed);
                        perf->les_proxy_submit_ns.fetch_add(submit_ns, std::memory_order_relaxed);
                    }
                }
                network.set_les_proxy_prediction(proxy_link.chain.target_moe,
                                                 std::move(proxy_indices),
                                                 std::move(proxy_weights),
                                                 std::move(prediction_ready));
                proxy_prediction_ready = network.get_les_proxy_state(proxy_link.chain.target_moe).prediction_ready;
                prediction_created = true;
                if (proxy_only) {
                    if (auto* perf = moe_otd::get_perf_counters()) {
                        perf->les_early_predictions.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }

        if (network.is_les_prefetch_enabled() && prediction_created && proxy_prediction_ready && active_proxy_link) {
            try {
                stream.flush();
                proxy_prediction_ready->wait();
                const auto proxy = network.get_les_proxy_state(active_proxy_link->chain.target_moe);
                std::vector<uint32_t> predicted_ids(std::min(active_proxy_link->top_k,
                                                            moe_otd::LesSpeculationPolicy::predict_k));
                cldnn::mem_lock<uint32_t, cldnn::mem_lock_type::read> indices(proxy.predicted_indices, stream);
                std::copy_n(indices.data(), predicted_ids.size(), predicted_ids.data());
                auto target = network.get_primitive(active_proxy_link->chain.target_moe);
                target->get_impl()->enqueue_les_prefetch(*target, predicted_ids, proxy_prediction_ready);
            } catch (const std::exception& error) {
                GPU_DEBUG_INFO << "[LES_PREDICT_PREFETCH] target=" << active_proxy_link->chain.target_moe
                               << " skipped=" << error.what() << std::endl;
            }
        }
        if (proxy_only) {
            return proxy_prediction_ready;
        }

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

        auto actual_routing_ready = stream.enqueue_kernel(*routing_stage->kernel,
                                                          kargs_desc,
                                                          {},
                                                          events,
                                                          instance.needs_completion_event());
        return actual_routing_ready;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MoeRouterFusedOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_router_fused>());
    return std::make_unique<MoeRouterFusedImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_router_fused)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoeRouterFusedImpl)
