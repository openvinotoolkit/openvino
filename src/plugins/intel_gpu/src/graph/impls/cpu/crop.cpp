// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "crop_inst.h"
#include "registry/implementation_map.hpp"

#include "openvino/op/slice.hpp"

namespace cldnn {
namespace cpu {

struct crop_impl : public typed_primitive_impl<crop> {
    using parent = typed_primitive_impl<crop>;
    using parent::parent;

    std::shared_ptr<ov::op::Op> op;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::crop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<crop_impl>(*this);
    }

    crop_impl() : parent("crop_cpu_impl") {}

    explicit crop_impl(const crop_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<crop>(), "[GPU] Incorrect program_node type");
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, crop_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        auto params = instance.get_impl_params();
        auto input_layout = params->input_layouts[0];
        auto input_offset = params->input_offsets[0];
        auto output_layout = params->output_layouts[0];

        auto input_shape = input_layout.get_partial_shape().to_shape();
        auto offsets_shape = input_offset.get_partial_shape(input_shape.size(), input_layout.get_rank()).to_shape();
        auto output_shape = output_layout.get_partial_shape().to_shape();

        OPENVINO_ASSERT(offsets_shape.size() == output_shape.size(), "[GPU] Offset shape is supposed to have the same rank as output shape");

        auto input_mem_ptr = instance.input_memory_ptr();
        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read> input_lock(input_mem_ptr, stream);
        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        auto padded_output = static_cast<bool>(output_mem_ptr->get_layout().data_padding);
        OPENVINO_ASSERT(!padded_output, "[GPU] Padded output is not supported yet");

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        std::vector<int64_t> start_vec(offsets_shape.begin(), offsets_shape.end());
        std::vector<int64_t> steps_vec(input_shape.size(), 1);
        std::vector<int64_t> stop_vec;

        for (size_t i = 0; i < start_vec.size(); i++)
            stop_vec.push_back(start_vec[i] + output_shape[i]);


        auto start_tensor = ov::Tensor(ov::element::i64, {start_vec.size()}, start_vec.data());
        auto stop_tensor = ov::Tensor(ov::element::i64, {stop_vec.size()}, stop_vec.data());
        auto steps_tensor = ov::Tensor(ov::element::i64, {steps_vec.size()}, steps_vec.data());

        auto input_tensor = make_tensor(params->input_layouts[0], input_lock.data());
        auto output_tensor = make_tensor(params->output_layouts[0], output_lock.data());

        input_host_tensors.push_back(input_tensor);
        input_host_tensors.push_back(start_tensor);
        input_host_tensors.push_back(stop_tensor);
        input_host_tensors.push_back(steps_tensor);

        output_host_tensors.push_back(output_tensor);

        if (!op)
            op = std::make_shared<ov::op::v8::Slice>();

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute crop primitive with id ", instance.id());

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const crop_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<crop_impl>();
    }
};


namespace detail {

attach_crop_impl::attach_crop_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<crop>::add(impl_types::cpu, shape_types::static_shape, crop_impl::create, types, formats);
    implementation_map<crop>::add(impl_types::cpu, shape_types::dynamic_shape, crop_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::crop_impl)
