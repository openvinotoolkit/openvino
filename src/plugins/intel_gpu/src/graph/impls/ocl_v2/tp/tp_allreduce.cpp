// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/ze_kernel.hpp"
#include "ze/ze_stream.hpp"
#include "ze/tp/rank_communicator.hpp"

#include "tp_allreduce.hpp"

#include <mutex>

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "impls/cpu/cpu_impl_helpers.hpp"
#include "tp_allreduce_inst.h"

namespace ov::intel_gpu::ocl {
namespace {

// The reduction step the TP channel launches between its copies. Only the
// element type specializes it: buffers, count and launch size are set by the
// channel when it records the collective.
class TPAllReduceGenerator : public KernelGenerator {
public:
    TPAllReduceGenerator() : KernelGenerator("tp_allreduce") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        switch (params.get_input_layout(0).data_type) {
        case data_types::bf16:
            jit.make("TP_BF16", 1);
            break;
        case data_types::f16:
            jit.make("TP_F16", 1);
            break;
        default:
            break;
        }
        return jit;
    }

    // The stream never launches this kernel, so it binds no primitive memory.
    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            kd.params.workGroups.global = {1, 1, 1};
            kd.params.workGroups.local = {1, 1, 1};
        }};
    }
};

ze_command_list_handle_t model_queue_of(cldnn::stream& s) {
    auto* ze = dynamic_cast<cldnn::ze::ze_stream*>(&s);
    OPENVINO_ASSERT(ze != nullptr, "[GPU] tp_allreduce expects the Level Zero stream");
    return ze->get_command_list().handle();
}

class TPAllReduceImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::TPAllReduceImpl)

    Stage::Ptr reduce = make_stage<TPAllReduceGenerator>();

    TPAllReduceImpl() : PrimitiveImplOCL(TPAllReduce::get_type_info_static()) {}
    TPAllReduceImpl(const program_node& node, const RuntimeParams& params) : TPAllReduceImpl() {
        add_stage(reduce, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<TPAllReduceImpl>(this);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        const auto& channel_ptr = instance.get_network().get_tp_channel();
        OPENVINO_ASSERT(channel_ptr != nullptr,
                        "[GPU] tp_allreduce has no TP channel; the model must be compiled or imported with a TP context");
        auto& channel = *channel_ptr;
        const bool async = channel.run_spliced();

        if (channel.options().verbosity >= ov::log::Level::INFO) {
            static std::once_flag reported;
            std::call_once(reported, [&] {
                cldnn::ze::tp::log_stream() << "[TP] collective rides "
                                            << (async ? "in the model queue" : "on its own queue (synchronous)")
                                            << ", in-order=" << (stream.get_queue_type() == QueueTypes::in_order) << std::endl;
            });
        }

        // Spliced, the recording lands in the model's in-order queue after the
        // operations that produced our input and before whatever reads our
        // output. Draining is the fallback: on the coordinator's own queue the
        // collective has no ordering against the model at all.
        if (!async) {
            if (!events.empty()) {
                stream.wait_for_events(events);
            }
            stream.finish();
        }

        const auto& layout = instance.get_impl_params()->get_input_layout(0);
        channel.allreduce(static_cast<int>(instance.get_typed_desc<tp_allreduce>()->collective_id),
                          instance.input_memory_ptr()->buffer_ptr(),
                          instance.output_memory_ptr()->buffer_ptr(),
                          layout.count(),
                          ov::element::Type{layout.data_type},
                          reduce_kernel(),
                          async ? model_queue_of(stream) : nullptr);
        return cldnn::cpu::make_output_event(stream, instance.is_output());
    }

private:
    ze_kernel_handle_t reduce_kernel() const {
        const auto* kernel = dynamic_cast<const cldnn::ze::ze_kernel*>(reduce->kernel.get());
        OPENVINO_ASSERT(kernel != nullptr, "[GPU] tp_allreduce has no Level Zero reduction kernel");
        return kernel->get_kernel_handle();
    }
};

}  // namespace

std::unique_ptr<primitive_impl> TPAllReduce::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<tp_allreduce>());
    return std::make_unique<TPAllReduceImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::tp_allreduce)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::TPAllReduceImpl)

#endif  // ENABLE_TP
