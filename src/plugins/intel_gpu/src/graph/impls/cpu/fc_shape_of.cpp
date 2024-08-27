// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"
#include "fc_shape_of_inst.h"
#include "impls/registry/implementation_map.hpp"
#include "matmul_shape_inference.hpp"

// #include "openvino/op/shape_of.hpp"

namespace cldnn {
namespace cpu {

struct fc_shape_of_impl : public typed_primitive_impl<fc_shape_of> {
    using parent = typed_primitive_impl<fc_shape_of>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::fc_shape_of_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fc_shape_of_impl>(*this);
    }

    fc_shape_of_impl() : parent("fc_shape_of_cpu_impl") {}

    explicit fc_shape_of_impl(const fc_shape_of_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<fc_shape_of>(), "[GPU] Incorrect program_node type");
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, fc_shape_of_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "fc_shape_of::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();

        auto output_mem_ptr = instance.output_memory_ptr();

        auto output_dt = instance.get_impl_params()->get_output_layout().data_type;

        auto input_layout = instance.get_input_layout(0);
        auto weights_layout = instance.get_input_layout(1);

        ov::op::v0::MatMul op;
        op.set_transpose_b(true);

        std::vector<ov::PartialShape> input_shapes = {
            input_layout.get<ov::PartialShape>(),
            weights_layout.get<ov::PartialShape>()
        };

        std::vector<ov::PartialShape> output_shapes = ov::op::v0::shape_infer(&op, input_shapes);
        ov::Shape output_shape = output_shapes[0].get_shape();

        auto data_layout = instance.get_input_layout(2);

        if ((output_shape[0] == 1 || output_shape[1] == 1) ||
            (data_layout.batch() == 1 && data_layout.feature() == 1) ||
            (ov::shape_size(output_shape) == data_layout.count())) {
            output_shape = data_layout.get_shape();
        }

        if (output_dt == data_types::i32) {
            cldnn::mem_lock<int32_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
            for (size_t i = 0; i < output_shape.size(); i++)
                output_lock[i] = static_cast<int32_t>(output_shape[i]);
        } else if (output_dt == data_types::i64) {
            cldnn::mem_lock<int64_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);
            for (size_t i = 0; i < output_shape.size(); i++)
                output_lock[i] = static_cast<int64_t>(output_shape[i]);
        } else {
            OPENVINO_THROW("[GPU] Couldn't execute shape_of operation: unsupported output data type (", output_dt , ")");
        }

        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }

        return stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const fc_shape_of_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<fc_shape_of_impl>();
    }
};


namespace detail {

attach_fc_shape_of_impl::attach_fc_shape_of_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<fc_shape_of>::add(impl_types::cpu, shape_types::any, fc_shape_of_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::fc_shape_of_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::fc_shape_of)
