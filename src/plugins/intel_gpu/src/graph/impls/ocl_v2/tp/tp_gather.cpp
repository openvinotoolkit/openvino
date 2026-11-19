// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/ze_stream.hpp"
#include "tp_gather_recorder.hpp"
#include "tp_common.hpp"

#include "tp_gather.hpp"

#include "../primitive_ocl_base.hpp"
#include "tp_gather_inst.h"

namespace ov::intel_gpu::ocl {
namespace {

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

        // A rank's slice -- everything from the gathered axis on -- is contiguous
        // in its own buffer but strided in the root's; rows is everything above
        // that axis flattened. The axis counts from the front, which also holds
        // for the 4D layouts static shapes are padded to.
        const auto shape = input_layout.get_shape();
        const auto rank_len = static_cast<int64_t>(shape.size());
        const int64_t axis = desc->axis;
        OPENVINO_ASSERT(axis >= 0 && axis < rank_len,
                        "[GPU][TP] tp_gather axis ",
                        desc->axis,
                        " is out of range for ",
                        input_layout.to_short_string());
        size_t rows = 1;
        for (int64_t i = 0; i < axis; ++i) {
            rows *= shape[static_cast<size_t>(i)];
        }
        const size_t slice_elems = input_layout.count() / std::max<size_t>(rows, 1);

        const auto& channel = require_channel(instance.get_network().get_tp_channel(), "tp_gather");
        const bool async = channel.run_spliced();
        drain_unless_spliced(stream, events, async);

        m_recorder.run(channel,
                       desc->collective_id,
                       instance.id(),
                       instance.input_memory_ptr()->buffer_ptr(),
                       instance.output_memory_ptr()->buffer_ptr(),
                       rows,
                       slice_elems,
                       ov::element::Type{input_layout.data_type},
                       async ? model_queue_of(stream, "tp_gather") : nullptr);
        return collective_event(stream, events, instance.is_output());
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
