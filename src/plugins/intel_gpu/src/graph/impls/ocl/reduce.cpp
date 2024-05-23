// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reduce_inst.h"
#include "reduce/reduce_kernel_selector.h"
#include "reduce/reduce_kernel_ref.h"

namespace cldnn {
namespace ocl {
namespace {
static std::vector<uint16_t> convert_axes(std::vector<int64_t> axes, size_t rank) {
    std::vector<uint16_t> converted_axes;
    for (auto axis : axes) {
        if (axis == 0 || axis == 1) {
            converted_axes.push_back(axis);
            continue;
        }

        if (axis < 0)
            axis = axis + rank;

        converted_axes.push_back(static_cast<uint16_t>(rank + 1 - axis));
    }

    return converted_axes;
}

kernel_selector::reduce_mode cldnn_2_reduce_mode(reduce_mode mode) {
    switch (mode) {
        case reduce_mode::max:
            return kernel_selector::reduce_mode::MAX;
        case reduce_mode::min:
            return kernel_selector::reduce_mode::MIN;
        case reduce_mode::mean:
            return kernel_selector::reduce_mode::MEAN;
        case reduce_mode::prod:
            return kernel_selector::reduce_mode::PROD;
        case reduce_mode::sum:
            return kernel_selector::reduce_mode::SUM;
        case reduce_mode::logical_and:
            return kernel_selector::reduce_mode::AND;
        case reduce_mode::logical_or:
            return kernel_selector::reduce_mode::OR;
        case reduce_mode::sum_square:
            return kernel_selector::reduce_mode::SUM_SQUARE;
        case reduce_mode::l1:
            return kernel_selector::reduce_mode::L1;
        case reduce_mode::l2:
            return kernel_selector::reduce_mode::L2;
        case reduce_mode::log_sum:
            return kernel_selector::reduce_mode::LOG_SUM;
        case reduce_mode::log_sum_exp:
            return kernel_selector::reduce_mode::LOG_SUM_EXP;
        default:
            assert(0);
            return kernel_selector::reduce_mode::MAX;
    }
}
}  // namespace
struct reduce_impl : typed_primitive_impl_ocl<reduce> {
    using parent = typed_primitive_impl_ocl<reduce>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reduce_kernel_selector;
    using kernel_params_t = kernel_selector::reduce_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reduce_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduce_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        auto& kernel_selector = kernel_selector_t::Instance();
        auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
        kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<reduce>();
        auto params = get_default_params<kernel_selector::reduce_params>(impl_param, is_shape_agnostic);

        params.reduceAxes = convert_axes(primitive->axes, impl_param.input_layouts[0].get_rank());
        params.keepDims = primitive->keep_dims;
        params.reduceMode = cldnn_2_reduce_mode(primitive->mode);
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<reduce>& instance) override {
        std::cout << "bell debug reduce kernel";
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return aggregate_events(events, stream, false, instance.is_output());
        }
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                        "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                        "[GPU] KernelData count: ", _kernel_data.kernels.size(), "\n",
                                                                        "[GPU] Likely some issue with empty tensor handling happened");
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution)
                continue;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernel_data.kernels[kd_idx].params;
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue kernel " << kd_idx << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;
            for (auto& iter : args.inputs) {
                std::cout << iter->count() << std::endl;
                std::cout << iter->get_allocation_type() << std::endl;
                std::cout << iter->buffer_ptr() << std::endl;
            }

            for (auto& iter : args.outputs) {
                std::cout << iter->count() << std::endl;
                std::cout << iter->get_allocation_type() << std::endl;
                std::cout << iter->buffer_ptr() << std::endl;
            }

            auto ev = stream.enqueue_kernel(*_kernels[kd_idx], params, args, tmp_events, needs_completion_event);
            std::cout << "bell debug reduce kernel enqueue finished" << std::endl;
            if (_kernel_data.needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return aggregate_events(tmp_events, stream);

        bool group_events = (all_events.size() > 1);
        return aggregate_events(all_events, stream, group_events);
                            }
};

namespace detail {

attach_reduce_impl::attach_reduce_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i8,
        data_types::u8
    };

    auto static_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv16
    };

    implementation_map<reduce>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<reduce>::create<reduce_impl>,
                                    types,
                                    static_formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx
    };

    implementation_map<reduce>::add(impl_types::ocl,
                                    shape_types::dynamic_shape,
                                    typed_primitive_impl_ocl<reduce>::create<reduce_impl>,
                                    types,
                                    dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reduce_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reduce)
