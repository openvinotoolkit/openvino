// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "tp_gather_inst.h"
#include "registry/implementation_map.hpp"

#include "tp_gpu/tp_device_coordinator.hpp"

namespace cldnn {
namespace ocl {

// Like tp_allreduce, an "OCL" impl that compiles no OpenCL kernel: the work is
// a strided cross-device copy issued through Level Zero by the coordinator.
// Registering as impl_types::ocl with is_cpu()=false is what makes intel_gpu
// place this primitive's buffers in usm_device.
struct tp_gather_impl : public typed_primitive_impl<tp_gather> {
    using parent = typed_primitive_impl<tp_gather>;
    using parent::parent;

    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 1;
    int64_t axis = -1;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::tp_gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<tp_gather_impl>(*this);
    }

    tp_gather_impl() : parent() {}

    explicit tp_gather_impl(const tp_gather_node& node) {
        auto desc = node.get_primitive();
        group_id = desc->group_id;
        collective_id = desc->collective_id;
        rank = desc->rank;
        world_size = desc->world_size;
        axis = desc->axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << group_id;
        ob << collective_id;
        ob << rank;
        ob << world_size;
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> group_id;
        ib >> collective_id;
        ib >> rank;
        ib >> world_size;
        ib >> axis;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            tp_gather_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        if (!events.empty()) {
            stream.wait_for_events(events);
        }
        stream.finish();

        auto params = instance.get_impl_params();
        const auto& input_layout = params->input_layouts[0];
        auto input_mem_ptr = instance.input_memory_ptr();
        auto output_mem_ptr = instance.output_memory_ptr();

        // The gathered axis is the innermost one for a vocabulary projection,
        // so a rank's slice is contiguous in its own buffer but strided in the
        // root's.  rows is everything above that axis flattened.
        const auto shape = input_layout.get_shape();
        const int64_t rank_len = static_cast<int64_t>(shape.size());
        const int64_t norm_axis = axis >= 0 ? axis : axis + rank_len;
        OPENVINO_ASSERT(norm_axis == rank_len - 1,
                        "[GPU] tp_gather currently gathers the innermost axis only, got ", axis);

        const size_t slice_elems = shape.back();
        const size_t rows = input_layout.count() / std::max<size_t>(slice_elems, 1);

        const auto etype = input_layout.data_type;
        ov::element::Type ov_dtype;
        if (etype == data_types::f16) {
            ov_dtype = ov::element::f16;
        } else if (etype == data_types::f32) {
            ov_dtype = ov::element::f32;
        } else {
            OPENVINO_THROW("[GPU] tp_gather: unsupported data type ", etype);
        }

        const auto& registry = instance.get_network().get_collective_comm_registry();
        OPENVINO_ASSERT(registry != nullptr,
            "[GPU] tp_gather requires a collective registry; the tensor-parallel plugin must inject "
            "one into the compiled model before inference");
        const auto& coordinator = registry->get_group(group_id);
        OPENVINO_ASSERT(coordinator != nullptr,
            "[GPU] tp_gather ocl impl requires TPDeviceCoordinator (shared L0 context)");

        coordinator->gather_to_root(static_cast<int>(collective_id),
                                    static_cast<int>(rank),
                                    input_mem_ptr->buffer_ptr(),
                                    output_mem_ptr->buffer_ptr(),
                                    rows,
                                    slice_elems,
                                    ov_dtype);
        return cpu::make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const tp_gather_node& arg,
                                                  const kernel_impl_params& impl_param) {
        return std::make_unique<tp_gather_impl>(arg);
    }
};

namespace detail {

attach_tp_gather_impl::attach_tp_gather_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    implementation_map<tp_gather>::add(impl_types::ocl,
                                       shape_types::any,
                                       tp_gather_impl::create,
                                       types,
                                       formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::tp_gather_impl)

#endif  // ENABLE_TP_GPU
