// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_primitive.hpp"

#include <memory>
#include <vector>

#include "intel_gpu/primitives/mlir_primitive.hpp"
#include "intel_gpu/runtime/tensor_accessor.hpp"   // cldnn::make_tensor
#include "mlir_primitive_inst.h"
#include "openvino/core/node.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "register.hpp"
#include "registry/implementation_map.hpp"

namespace cldnn::common {

struct mlir_primitive_impl : typed_primitive_impl<mlir_primitive> {
    using parent = typed_primitive_impl<mlir_primitive>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::common::mlir_primitive_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<mlir_primitive_impl>(*this);
    }

    mlir_primitive_impl() : parent() {}

    explicit mlir_primitive_impl(const mlir_primitive_node& outer) { set_node_params(outer); }

    void set_node_params(const program_node& /*arg*/) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& dependent_events,
                            mlir_primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        const auto& prim = instance.node->get_primitive();
        const auto& op = prim->op;

        ov::TensorVector input_gpu_tensors;
        ov::TensorVector output_gpu_tensors;
        std::vector<bool> is_usm_ptr;
        input_gpu_tensors.reserve(instance.inputs_memory_count());
        output_gpu_tensors.reserve(instance.outputs_memory_count());
        is_usm_ptr.reserve(instance.inputs_memory_count() + instance.outputs_memory_count());

        auto process_buffer = [&stream, &is_usm_ptr](memory::ptr mem, ov::TensorVector& tensors) {
            switch (mem->get_allocation_type()) {
                case allocation_type::cl_mem: {
                    if (void* cl_buff = mem->get_handle()) {
                        tensors.push_back(make_tensor(mem->get_layout(), cl_buff));
                        is_usm_ptr.push_back(false);
                    } else {
                        OPENVINO_THROW("Memory handle is null for cl_mem");
                    }
                    break;
                }
                case allocation_type::usm_host:
                case allocation_type::usm_shared:
                case allocation_type::usm_device: {
                    auto usm_ptr = mem->buffer_ptr();
                    // Seems to only occur with Out-Of-Order queues sometimes. Can't reproduce this anymore, uncomment if needed.
                    // HACK: force move to device, can we do better than this?
                    // auto gpu_buff = dynamic_cast<cldnn::ocl::gpu_usm*>(mem.get());
                    // auto& usm_helper = gpu_buff->get_buffer().getUsmHelper();
                    // usm_helper.enqueue_memcpy(
                    //     dynamic_cast<cldnn::ocl::ocl_stream&>(stream).get_cl_queue(),
                    //     usm_ptr,
                    //     usm_ptr,
                    //     mem->get_layout().bytes_count());
                    tensors.push_back(make_tensor(mem->get_layout(), usm_ptr));
                    is_usm_ptr.push_back(true);
                    break;
                }
                default:
                    OPENVINO_THROW("Unsupported memory type");
            }
        };

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            process_buffer(instance.input_memory_ptr(i), input_gpu_tensors);
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            process_buffer(instance.output_memory_ptr(i), output_gpu_tensors);
        }

        ov::EvaluationContext meta;
        if (void* queue = stream.get_handle()) {
            meta.insert(ov::intel_gpu::ocl_queue(queue));
        } else {
            OPENVINO_THROW("Unsupported queue type");
        }
        meta.insert(ov::internal::mlir_meta::is_kernel_arg_usm(is_usm_ptr));

        std::vector<void*> events_list;
        cl_event* result_event = nullptr;
        if (stream.get_queue_type() == QueueTypes::out_of_order) {
            events_list.reserve(dependent_events.size() + 1);
            for (auto& ev : dependent_events) {
                if (void* cl_ev = ev->get_handle()) {
                    events_list.push_back(cl_ev);
                } else {
                    OPENVINO_THROW("Unsupported event type");
                }
            }
            meta.insert(ov::internal::mlir_meta::wait_list(events_list));
            // 'cl_event' is a pointer itself, that's why we pass pointer to a pointer here.
            meta.insert(ov::internal::mlir_meta::result_event(reinterpret_cast<void**>(result_event)));
        }

        OPENVINO_ASSERT(op->evaluate(
                        output_gpu_tensors, input_gpu_tensors, meta),
                        "[GPU] Couldn't execute MLIROp ", op->get_friendly_name());

        event::ptr ev;
        if (stream.get_queue_type() == QueueTypes::out_of_order) {
            OPENVINO_ASSERT(result_event != nullptr, "Result cl_event is not set");
            ev = stream.create_base_event(*result_event);
        } else {
            ev = stream.create_user_event(true);
        }

        return ev;
    }

    static std::unique_ptr<primitive_impl> create(const mlir_primitive_node& arg,
                                                  const kernel_impl_params& /*params*/) {
        return std::make_unique<mlir_primitive_impl>(arg);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    void save(BinaryOutputBuffer& ob) const override { parent::save(ob); }
    void load(BinaryInputBuffer& ib) override { parent::load(ib); }

    bool is_cpu() const override { return false; }
};

std::unique_ptr<primitive_impl> MLIRPrimitiveImplementationManager::create_impl(
        const program_node& node,
        const kernel_impl_params& params) const {
    assert(node.is_type<mlir_primitive>());
    return mlir_primitive_impl::create(static_cast<const mlir_primitive_node&>(node), params);
}

namespace detail {

attach_mlir_primitive_common::attach_mlir_primitive_common() {
    implementation_map<mlir_primitive>::add(impl_types::common,
                                            shape_types::dynamic_shape,
                                            mlir_primitive_impl::create,
                                            {},
                                            {});
    implementation_map<mlir_primitive>::add(impl_types::common, mlir_primitive_impl::create, {});
}

}  // namespace detail

}  // namespace cldnn::common

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::common::mlir_primitive_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mlir_primitive)
