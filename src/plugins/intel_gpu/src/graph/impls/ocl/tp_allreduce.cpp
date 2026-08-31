// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

// The collective runs on Level Zero and so does the runtime underneath this
// impl: ENABLE_TP_GPU is rejected at configure time unless GPU_RT_TYPE=ZE
// (cmake/features.cmake).  That is what lets the two share a queue instead of
// the collective draining the model's, and it is why this private runtime
// header is included from an impl at all.
//
// It has to come first: zero_api.hpp undefines its symbols_list macro on the
// way out unless the includer asked to keep it, and ze_common.hpp is the one
// that asks.  Let the coordinator's header include it first and ze_common
// finds the macro already gone.
#include "ze/ze_stream.hpp"

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "tp_allreduce_inst.h"
#include "registry/implementation_map.hpp"

#include "tp_gpu/tp_device_coordinator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace cldnn {
namespace ocl {

namespace {

// The immediate command list intel_gpu runs the model on.
ze_command_list_handle_t model_queue_of(stream& s) {
    auto* ze = dynamic_cast<cldnn::ze::ze_stream*>(&s);
    OPENVINO_ASSERT(ze != nullptr, "[GPU] tp_allreduce expects the Level Zero stream");
    return ze->get_queue();
}

}  // namespace

// "OCL" impl that does not actually compile an OpenCL kernel.  All work is
// dispatched via Level Zero by TPDeviceCoordinator.  Registering as
// impl_types::ocl with is_cpu()=false makes intel_gpu allocate IO buffers
// in usm_device, which is the only way to get device-local bandwidth on
// the all-reduce hot path.
struct tp_allreduce_impl : public typed_primitive_impl<tp_allreduce> {
    using parent = typed_primitive_impl<tp_allreduce>;
    using parent::parent;

    uint32_t group_id = 0;
    uint32_t collective_id = 0;
    uint32_t rank = 0;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::tp_allreduce_impl)

    bool is_cpu() const override { return false; }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<tp_allreduce_impl>(*this);
    }

    tp_allreduce_impl() : parent("tp_allreduce_ocl_impl") {}

    explicit tp_allreduce_impl(const tp_allreduce_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<tp_allreduce>(), "[GPU] Incorrect program_node type");
        const auto& prim = arg.as<tp_allreduce>().get_primitive();
        group_id = prim->group_id;
        collective_id = prim->collective_id;
        rank = prim->rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << group_id;
        ob << collective_id;
        ob << rank;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> group_id;
        ib >> collective_id;
        ib >> rank;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            tp_allreduce_inst& instance) override {
        auto& stream = instance.get_network().get_stream();

        // Handing the collective to the model's queue instead of draining the
        // stream and running it on our own.  The queue is in-order, so the
        // spliced recording lands after the operations that produced our
        // input and before whatever reads our output -- `events` and
        // stream.finish() were both only ever standing in for that.
        const auto& coordinator = coordinator_of(instance);
        const bool async = coordinator->async_supported() &&
                           std::getenv("TP_SYNC_COLLECTIVE") == nullptr;

        if (std::getenv("TP_PROF") != nullptr) {
            static std::once_flag reported;
            std::call_once(reported, [&] {
                std::cerr << "[TP] collective rides "
                          << (async ? "in the model queue" : "on its own queue (synchronous)")
                          << ", in-order=" << (stream.get_queue_type() == QueueTypes::in_order)
                          << std::endl;
            });
        }

        if (!async) {
            if (!events.empty()) {
                stream.wait_for_events(events);
            }
            run_synchronously(instance, stream);
            return cpu::make_output_event(stream, instance.is_output());
        }

        auto [in_dev, out_dev, num_elements, ov_dtype] = collective_operands(instance);
        coordinator->allreduce_async(static_cast<int>(collective_id),
                                     static_cast<int>(rank),
                                     in_dev, out_dev, num_elements, ov_dtype,
                                     model_queue_of(stream));
        return cpu::make_output_event(stream, instance.is_output());
    }

    /// The original path: drain this rank's stream, then run the collective on
    /// the coordinator's own queues and wait for it.  Kept for A/B against the
    /// spliced path and as the fallback when the driver has no splice.
    void run_synchronously(tp_allreduce_inst& instance, stream& stream) {

        // This call is the seam in the stretch between two collectives:
        // everything before it is the host dispatching the model's operations,
        // and the call itself is the wait for this rank's GPU to catch up.
        // The ranks arrive at the next collective ~90 us apart and the group
        // moves at the speed of the last one, so which side of this seam the
        // spread comes from decides whether anything can be done about it.
        static const bool skew_enabled = std::getenv("TP_SKEW") != nullptr;
        const auto t_fin = std::chrono::steady_clock::now();
        stream.finish();
        if (skew_enabled) {
            constexpr int kMaxRanks = 16;
            // One thread per rank is inside a collective at a time, so plain
            // counters indexed by rank race with nothing.
            static uint64_t fin_calls[kMaxRanks]{};
            static uint64_t fin_sum[kMaxRanks]{};
            static uint64_t fin_min[kMaxRanks]{};
            static uint64_t fin_max[kMaxRanks]{};
            const auto dt = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_fin)
                    .count());
            const int r = static_cast<int>(rank) % kMaxRanks;
            fin_min[r] = fin_calls[r] == 0 ? dt : std::min(fin_min[r], dt);
            fin_max[r] = std::max(fin_max[r], dt);
            fin_sum[r] += dt;
            if ((++fin_calls[r] % 6400) == 0 && r == 0) {
                for (int i = 0; i < kMaxRanks; ++i) {
                    if (fin_calls[i] == 0) {
                        continue;
                    }
                    std::cerr << "[TP][SKEW] stream.finish rank " << i << " over "
                              << fin_calls[i] << " calls: mean="
                              << (static_cast<double>(fin_sum[i]) / 1.0e3 /
                                  static_cast<double>(fin_calls[i]))
                              << "us min=" << (static_cast<double>(fin_min[i]) / 1.0e3)
                              << "us max=" << (static_cast<double>(fin_max[i]) / 1.0e3) << "us"
                              << std::endl;
                }
            }
        }

        auto [in_dev, out_dev, num_elements, ov_dtype] = collective_operands(instance);
        coordinator_of(instance)->allreduce(static_cast<int>(collective_id),
                                            static_cast<int>(rank),
                                            in_dev, out_dev, num_elements, ov_dtype);
    }

    /// The coordinator is runtime state of the network, injected by the
    /// tensor-parallel plugin after this model was compiled or imported.
    const ov::tp_gpu::TPDeviceCoordinatorPtr& coordinator_of(tp_allreduce_inst& instance) const {
        const auto& registry = instance.get_network().get_collective_comm_registry();
        OPENVINO_ASSERT(registry != nullptr,
            "[GPU] tp_allreduce requires a collective registry; the tensor-parallel plugin must inject "
            "one into the compiled model before inference");
        const auto& coordinator = registry->get_group(group_id);
        OPENVINO_ASSERT(coordinator != nullptr,
            "[GPU] tp_allreduce ocl impl requires TPDeviceCoordinator (shared L0 context)");
        return coordinator;
    }

    struct Operands {
        void* in_dev;
        void* out_dev;
        size_t num_elements;
        ov::element::Type dtype;
    };

    static Operands collective_operands(tp_allreduce_inst& instance) {
        const auto& input_layout = instance.get_impl_params()->input_layouts[0];
        const auto etype = input_layout.data_type;

        ov::element::Type ov_dtype;
        if (etype == data_types::f16) {
            ov_dtype = ov::element::f16;
        } else if (etype == data_types::f32) {
            ov_dtype = ov::element::f32;
        } else {
            OPENVINO_THROW("[GPU] tp_allreduce: unsupported data type ", etype);
        }

        return {instance.input_memory_ptr()->buffer_ptr(),
                instance.output_memory_ptr()->buffer_ptr(),
                input_layout.count(),
                ov_dtype};
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

    implementation_map<tp_allreduce>::add(impl_types::ocl, shape_types::static_shape,
                                          tp_allreduce_impl::create, types, formats);
    implementation_map<tp_allreduce>::add(impl_types::ocl, shape_types::dynamic_shape,
                                          tp_allreduce_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::tp_allreduce_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::tp_allreduce)

#endif  // ENABLE_TP_GPU
