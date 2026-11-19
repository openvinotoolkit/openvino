// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/histc.hpp"

#include <vector>

#include "histc_inst.h"
#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace cpu {

// Calls f(i, offset) with the buffer offset of each logical element i of a plain, possibly padded layout.
template <class F>
void for_each_element(const layout& l, F&& f) {
    const auto dims = l.get_dims();
    const auto pitches = l.get_pitches();
    const auto base = l.get_linear_offset();
    for (size_t i = 0, n = l.count(); i < n; ++i) {
        size_t offset = base;
        size_t rest = i;
        for (size_t d = dims.size(); d > 0; --d) {
            const auto dim = static_cast<size_t>(dims[d - 1]);
            offset += rest % dim * static_cast<size_t>(pitches[d - 1]);
            rest /= dim;
        }
        f(i, offset);
    }
}

// Integer Histc bins in double, so devices without fp64 run it on the host.
struct histc_impl : public typed_primitive_impl<histc> {
    using parent = typed_primitive_impl<histc>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::histc_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<histc_impl>(*this);
    }

    histc_impl() : parent("histc_cpu_impl") {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, histc_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "histc::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();
        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        const auto* params = instance.get_impl_params();
        const auto desc = params->typed_desc<histc>();
        const auto& in_layout = params->get_input_layout(0);
        const auto& out_layout = params->get_output_layout(0);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> in_lock(instance.dep_memory_ptr(0), stream);
        cldnn::mem_lock<uint8_t, mem_lock_type::read_write> out_lock(instance.output_memory_ptr(), stream);

        const auto run = [&](const auto* in, auto* out) {
            using TIn = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
            using TOut = std::remove_pointer_t<decltype(out)>;
            std::vector<TIn> in_packed;
            if (in_layout.data_padding) {
                in_packed.resize(in_layout.count());
                for_each_element(in_layout, [&](size_t i, size_t offset) {
                    in_packed[i] = in[offset];
                });
                in = in_packed.data();
            }
            std::vector<TOut> out_packed(out_layout.count());
            ov::reference::histc(in, out_packed.data(), in_layout.count(), out_packed.size(), desc->min, desc->max);
            for_each_element(out_layout, [&](size_t i, size_t offset) {
                out[offset] = out_packed[i];
            });
        };
        const auto run_in = [&](const auto* in) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
            if (out_layout.data_type == data_types::i32) {
                run(in, reinterpret_cast<int32_t*>(out_lock.data()));
            } else {
                run(in, reinterpret_cast<T*>(out_lock.data()));
            }
        };
        switch (in_layout.data_type) {
        case data_types::i32:
            run_in(reinterpret_cast<const int32_t*>(in_lock.data()));
            break;
        case data_types::i8:
            run_in(reinterpret_cast<const int8_t*>(in_lock.data()));
            break;
        case data_types::u8:
            run_in(reinterpret_cast<const uint8_t*>(in_lock.data()));
            break;
        default:
            OPENVINO_THROW("[GPU] Unsupported input type ", in_layout.data_type, " of histc primitive with id ", instance.id());
        }

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const histc_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<histc_impl>();
    }
};

namespace detail {

attach_histc_impl::attach_histc_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    auto types = {
        data_types::i32,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<histc>::add(impl_types::cpu, shape_types::static_shape, histc_impl::create, types, formats);
    implementation_map<histc>::add(impl_types::cpu, shape_types::dynamic_shape, histc_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::histc_impl)
