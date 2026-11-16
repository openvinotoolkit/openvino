// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/ze_kernel.hpp"
#include "ze/ze_stream.hpp"
#include "tp_allreduce_recorder.hpp"
#include "tp_common.hpp"

#include "tp_allreduce.hpp"

#include <mutex>

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "tp_allreduce_inst.h"

namespace ov::intel_gpu::ocl {
namespace {

// The reduction step the recorder launches between its copies. The default jit
// constants type it by the input; buffers, count and launch size are set when
// the collective is recorded.
class TPAllReduceGenerator : public KernelGenerator {
public:
    TPAllReduceGenerator() : KernelGenerator("tp_allreduce") {}

protected:
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
        const auto& channel = require_channel(instance.get_network().get_tp_channel(), "tp_allreduce");
        const bool async = channel.run_spliced();

        if (channel.options().verbosity >= ov::log::Level::INFO) {
            static std::once_flag reported;
            std::call_once(reported, [&] {
                cldnn::ze::tp::LogLine{}.stream() << "[TP] collective rides "
                                            << (async ? "in the model queue" : "on its own queue (synchronous)")
                                            << ", in-order=" << (stream.get_queue_type() == QueueTypes::in_order) << std::endl;
            });
        }

        drain_unless_spliced(stream, events, async);

        const auto& layout = instance.get_impl_params()->get_input_layout(0);
        const auto collective_id = instance.get_typed_desc<tp_allreduce>()->collective_id;
        m_recorder.run(channel,
                       collective_id,
                       instance.id(),
                       instance.input_memory_ptr()->buffer_ptr(),
                       instance.output_memory_ptr()->buffer_ptr(),
                       layout.count(),
                       ov::element::Type{layout.data_type},
                       reduce_kernel(),
                       async ? model_queue_of(stream, "tp_allreduce") : nullptr);

        if (channel.options().validate_allreduce_output) {
            stream.finish();
            // One copy: reading device memory element by element crosses the bus every time.
            std::vector<uint8_t> output(instance.get_output_layout().bytes_count());
            instance.output_memory_ptr()->copy_to(stream, output.data(), 0, 0, output.size(), true);
            uint64_t checksum = 14695981039346656037ULL;  // FNV-1a
            for (const auto byte : output) {
                checksum = (checksum ^ byte) * 1099511628211ULL;
            }
            channel.group()->check_output(channel.rank(), collective_id, checksum, instance.id());
        }
        return collective_event(stream, events, instance.is_output());
    }

private:
    ze_kernel_handle_t reduce_kernel() const {
        const auto* kernel = dynamic_cast<const cldnn::ze::ze_kernel*>(reduce->kernel.get());
        OPENVINO_ASSERT(kernel != nullptr, "[GPU] tp_allreduce has no Level Zero reduction kernel");
        return kernel->get_kernel_handle();
    }

    TPAllReduceRecorder m_recorder;
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
