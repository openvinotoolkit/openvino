// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "scaled_shifted_clamp_experimental_inst.h"
#include "scaled_shifted_clamp_experimental/scaled_shifted_clamp_experimental_kernel_ref.h"
#include "scaled_shifted_clamp_experimental/scaled_shifted_clamp_experimental_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct scaled_shifted_clamp_experimental_impl : typed_primitive_impl_ocl<scaled_shifted_clamp_experimental> {
    using parent = typed_primitive_impl_ocl<scaled_shifted_clamp_experimental>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::scaled_shifted_clamp_experimental_kernel_selector;
    using kernel_params_t   = kernel_selector::scaled_shifted_clamp_experimental_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::scaled_shifted_clamp_experimental_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<scaled_shifted_clamp_experimental_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& ks   = kernel_selector_t::Instance();
            auto  impl = ks.GetImplementation(_kernel_data.kernelName);
            impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto        params = get_default_params<kernel_params_t>(impl_param, is_shape_agnostic);
        const auto  desc   = impl_param.typed_desc<scaled_shifted_clamp_experimental>();
        params.scale = desc->scale;
        params.bias  = desc->bias;
        params.lo    = desc->lo;
        params.hi    = desc->hi;
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }
        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_scaled_shifted_clamp_experimental_impl::attach_scaled_shifted_clamp_experimental_impl() {
    auto types   = {data_types::f16, data_types::f32};
    auto formats = {format::bfyx};
    implementation_map<scaled_shifted_clamp_experimental>::add(
        impl_types::ocl,
        shape_types::any,
        typed_primitive_impl_ocl<scaled_shifted_clamp_experimental>::create<scaled_shifted_clamp_experimental_impl>,
        types,
        formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scaled_shifted_clamp_experimental_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_shifted_clamp_experimental)
