// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "bevpool_v2/bevpool_v2_kernel_base.h"
#include "bevpool_v2/bevpool_v2_kernel_selector.h"
#include "bevpool_v2_inst.h"

namespace cldnn {
namespace ocl {

struct bevpool_v2_impl : typed_primitive_impl_ocl<bevpool_v2> {
    using parent = typed_primitive_impl_ocl<bevpool_v2>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::bevpool_v2_kernel_selector;
    using kernel_params_t = kernel_selector::bevpool_v2_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::bevpool_v2_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<bevpool_v2_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, bevpool_v2_inst& instance) override {
        stream& stream = instance.get_network().get_stream();
        // Ref kernel writes only positions addressed by interval descriptors and expects the rest to be zero.
        // Clear output buffer before execution to match reference behavior and avoid stale data from memory pool reuse.
        stream.enqueue_barrier();
        auto output_evt = instance.output_memory(0).fill(stream, {}, false);
        std::vector<event::ptr> ext_events(events);
        ext_events.push_back(output_evt);
        return parent::execute_impl(ext_events, instance);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<bevpool_v2>();
        auto params = get_default_params<kernel_selector::bevpool_v2_params>(impl_param, shape_agnostic);

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.input_channels = primitive->input_channels;
        params.output_channels = primitive->output_channels;
        params.image_width = primitive->image_width;
        params.image_height = primitive->image_height;
        params.feature_width = primitive->feature_width;
        params.feature_height = primitive->feature_height;

        params.x_bound_min = primitive->x_bound.min;
        params.x_bound_max = primitive->x_bound.max;
        params.x_bound_step = primitive->x_bound.step;

        params.y_bound_min = primitive->y_bound.min;
        params.y_bound_max = primitive->y_bound.max;
        params.y_bound_step = primitive->y_bound.step;

        params.z_bound_min = primitive->z_bound.min;
        params.z_bound_max = primitive->z_bound.max;
        params.z_bound_step = primitive->z_bound.step;

        params.d_bound_min = primitive->d_bound.min;
        params.d_bound_max = primitive->d_bound.max;
        params.d_bound_step = primitive->d_bound.step;

        return params;
    }
};

namespace detail {

attach_bevpool_v2_impl::attach_bevpool_v2_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32, data_types::i64};
    auto formats = {format::bfyx};
    implementation_map<bevpool_v2>::add(impl_types::ocl,
                                        shape_types::any,
                                        typed_primitive_impl_ocl<bevpool_v2>::create<bevpool_v2_impl>,
                                        types,
                                        formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::bevpool_v2_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::bevpool_v2)
