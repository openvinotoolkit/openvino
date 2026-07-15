// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TENSOR_PARALLEL

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "tp_allreduce_inst.h"
#include "registry/implementation_map.hpp"

#include "tensor_parallel/tp_coordination.hpp"
#include "tensor_parallel/tp_device_coordinator.hpp"

#include <cstring>
#include <iostream>
#include <mutex>

namespace cldnn {
namespace cpu {

struct tp_allreduce_impl : public typed_primitive_impl<tp_allreduce> {
    using parent = typed_primitive_impl<tp_allreduce>;
    using parent::parent;

    uint32_t collective_id = 0;
    uint32_t rank = 0;
    std::shared_ptr<ov::tp::TPCoordination> coordination;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::tp_allreduce_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<tp_allreduce_impl>(*this);
    }

    tp_allreduce_impl() : parent("tp_allreduce_cpu_impl") {}

    explicit tp_allreduce_impl(const tp_allreduce_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<tp_allreduce>(), "[GPU] Incorrect program_node type");
        const auto& prim = arg.as<tp_allreduce>().get_primitive();
        collective_id = prim->collective_id;
        rank = prim->rank;
        coordination = prim->coordination;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            tp_allreduce_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        // Full GPU synchronization — guarantees all preceding GPU work
        // (and its memory writes) are complete before we read on the CPU.
        // wait_for_events alone is insufficient: on some drivers the
        // event may signal before cache/TLB flushes finish, causing
        // sporadic stale reads.
        if (!events.empty()) {
            stream.wait_for_events(events);
        }
        stream.finish();

        auto params = instance.get_impl_params();
        auto input_layout = params->input_layouts[0];
        auto output_layout = params->output_layouts[0];

        auto input_mem_ptr = instance.input_memory_ptr();
        auto output_mem_ptr = instance.output_memory_ptr();

        size_t num_bytes = input_layout.bytes_count();
        size_t num_elements = input_layout.count();
        auto etype = input_layout.data_type;

        // ---- Device fast path ----
        // When the model was compiled under TENSOR_PARALLEL with a shared
        // L0 context, the coordination object carries a TPDeviceCoordinator
        // capable of running the collective directly on USM-device buffers
        // via the rank's L0 queue.  Both input and output must be USM-device
        // for the fast path; otherwise we fall back to the CPU staged path.
        if (auto dc = coordination->device_coordinator()) {
            const auto in_alloc  = input_mem_ptr->get_allocation_type();
            const auto out_alloc = output_mem_ptr->get_allocation_type();
            // tp_allreduce is registered as a CPU primitive, so intel_gpu
            // typically allocates its IO in usm_host (host+device accessible).
            // In the TP shared L0 context usm_host is reachable by every
            // rank's device, so the device fast-path accepts both
            // usm_device and usm_host.
            auto is_usm = [](allocation_type t) {
                return t == allocation_type::usm_device ||
                       t == allocation_type::usm_host ||
                       t == allocation_type::usm_shared;
            };
            static std::once_flag once;
            std::call_once(once, [&]() {
                std::cerr << "[TP][allreduce] first call: in_alloc=" << in_alloc
                          << " out_alloc=" << out_alloc
                          << " etype=" << etype << " n=" << num_elements
                          << " device_path=" << (is_usm(in_alloc) && is_usm(out_alloc) ? "yes" : "no")
                          << std::endl;
            });
            if (is_usm(in_alloc) && is_usm(out_alloc)) {
                ov::element::Type ov_dtype;
                if (etype == data_types::f16) {
                    ov_dtype = ov::element::f16;
                } else if (etype == data_types::f32) {
                    ov_dtype = ov::element::f32;
                } else {
                    OPENVINO_THROW("[GPU] tp_allreduce: unsupported data type ", etype);
                }
                void* in_dev  = input_mem_ptr->buffer_ptr();
                void* out_dev = output_mem_ptr->buffer_ptr();
                dc->allreduce(static_cast<int>(collective_id),
                              static_cast<int>(rank),
                              in_dev, out_dev, num_elements, ov_dtype);
                return make_output_event(stream, instance.is_output());
            }
        }

        // ---- CPU staged fallback ----
        cldnn::mem_lock<uint8_t, mem_lock_type::read> input_lock(input_mem_ptr, stream);
        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        // 1. Deposit this rank's data into the shared coordination buffer.
        coordination->deposit(collective_id, rank, input_lock.data(), num_bytes);

        // 2. Barrier: wait for all ranks to deposit.
        coordination->barrier_wait(collective_id);

        // 3. Sum all ranks' buffers into output.
        int world_size = coordination->world_size();
        auto* dst = output_lock.data();

        if (etype == data_types::f32) {
            auto* out = reinterpret_cast<float*>(dst);
            const auto* src0 = reinterpret_cast<const float*>(coordination->get_buffer(collective_id, 0));
            std::memcpy(out, src0, num_bytes);
            for (int r = 1; r < world_size; ++r) {
                const auto* src = reinterpret_cast<const float*>(coordination->get_buffer(collective_id, r));
                for (size_t i = 0; i < num_elements; ++i) {
                    out[i] += src[i];
                }
            }
        } else if (etype == data_types::f16) {
            // f16: promote to float for accumulation, store back as f16
            auto* out = reinterpret_cast<ov::float16*>(dst);
            const auto* src0 = reinterpret_cast<const ov::float16*>(coordination->get_buffer(collective_id, 0));
            std::memcpy(out, src0, num_bytes);
            for (int r = 1; r < world_size; ++r) {
                const auto* src = reinterpret_cast<const ov::float16*>(coordination->get_buffer(collective_id, r));
                for (size_t i = 0; i < num_elements; ++i) {
                    float val = static_cast<float>(out[i]) + static_cast<float>(src[i]);
                    out[i] = ov::float16(val);
                }
            }
        } else {
            OPENVINO_THROW("[GPU] tp_allreduce: unsupported data type ", etype);
        }

        // 4. Second barrier: ensure all ranks finished reading before next collective
        //    reuses the buffers.
        coordination->barrier_wait(collective_id);

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const tp_allreduce_node& arg,
                                                  const kernel_impl_params& impl_param) {
        return std::make_unique<tp_allreduce_impl>();
    }
};

namespace detail {

attach_tp_allreduce_impl::attach_tp_allreduce_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    implementation_map<tp_allreduce>::add(impl_types::cpu, shape_types::static_shape,
                                         tp_allreduce_impl::create, types, formats);
    implementation_map<tp_allreduce>::add(impl_types::cpu, shape_types::dynamic_shape,
                                         tp_allreduce_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::tp_allreduce_impl)

#endif  // ENABLE_TENSOR_PARALLEL
