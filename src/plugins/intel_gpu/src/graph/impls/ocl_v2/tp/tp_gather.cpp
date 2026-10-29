// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/ze_stream.hpp"
#include "tp_gather_recorder.hpp"

#include "tp_gather.hpp"

#include "../primitive_ocl_base.hpp"
#include "tp_gather_inst.h"

namespace ov::intel_gpu::ocl {
namespace {

ze_command_list_handle_t model_queue_of(cldnn::stream& s) {
    auto* ze = dynamic_cast<cldnn::ze::ze_stream*>(&s);
    OPENVINO_ASSERT(ze != nullptr, "[GPU] tp_gather expects the Level Zero stream");
    return ze->get_command_list().handle();
}

// No kernel: the work is copies issued through Level Zero by the recorder.
// Being an OCL impl (not a CPU one) is what places the buffers in usm_device.
class TPGatherImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::TPGatherImpl)

    TPGatherImpl() : PrimitiveImplOCL(TPGather::get_type_info_static()) {}
    TPGatherImpl(const program_node& node, const RuntimeParams& params) : TPGatherImpl() {}

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<TPGatherImpl>(this);
    }

    // No stages, so nothing for the kernels cache to provide or remember.
    void init_kernels(const cldnn::kernels_cache&, const RuntimeParams&) override {}
    void init_by_cached_kernels(const cldnn::kernels_cache&, std::vector<std::string>&) override {}
    std::vector<std::string> get_cached_kernel_ids(const cldnn::kernels_cache&) override {
        return {};
    }
    void set_kernels(cldnn::kernels_cache::compiled_kernels) override {}

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        const auto desc = instance.get_typed_desc<tp_gather>();
        const auto& input_layout = instance.get_impl_params()->get_input_layout(0);

        // The gathered axis is the innermost one for a vocabulary projection,
        // so a rank's slice is contiguous in its own buffer but strided in the
        // root's. rows is everything above that axis flattened.
        const auto shape = input_layout.get_shape();
        const auto rank_len = static_cast<int64_t>(shape.size());
        const int64_t axis = desc->axis >= 0 ? desc->axis : desc->axis + rank_len;
        OPENVINO_ASSERT(axis == rank_len - 1, "[GPU] tp_gather currently gathers the innermost axis only, got ", desc->axis);
        const size_t slice_elems = shape.back();
        const size_t rows = input_layout.count() / std::max<size_t>(slice_elems, 1);

        const auto& channel = instance.get_network().get_tp_channel();
        OPENVINO_ASSERT(channel != nullptr,
                        "[GPU] tp_gather has no TP channel; the model must be compiled or imported with a TP context");

        // Spliced, the recording lands in the model's in-order queue after the
        // projection that produced our slice. Draining is the fallback: on the
        // group's own queue the copy has no ordering against the model at all.
        const bool async = channel->run_spliced();
        if (!async) {
            if (!events.empty()) {
                stream.wait_for_events(events);
            }
            stream.finish();
        }

        m_recorder.run(*channel,
                       static_cast<int>(desc->collective_id),
                       instance.input_memory_ptr()->buffer_ptr(),
                       instance.output_memory_ptr()->buffer_ptr(),
                       rows,
                       slice_elems,
                       ov::element::Type{input_layout.data_type},
                       async ? model_queue_of(stream) : nullptr);
        // In-order stream: only a network output needs an event that completes with the copy.
        if (instance.is_output()) {
            return stream.enqueue_marker({}, true);
        }
        return stream.aggregate_events(events, false, false);
    }

private:
    TPGatherRecorder m_recorder;
};

}  // namespace

std::unique_ptr<primitive_impl> TPGather::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<tp_gather>());
    return std::make_unique<TPGatherImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::tp_gather)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::TPGatherImpl)

#endif  // ENABLE_TP
