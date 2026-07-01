// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "histc_inst.hpp"
#include "histc/histc_kernel_ref.hpp"
#include "histc/histc_kernel_selector.hpp"

namespace cldnn {
namespace ocl {

struct histc_impl : typed_primitive_impl_ocl<histc> {
    using parent = typed_primitive_impl_ocl<histc>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::histc_kernel_selector;
    using kernel_params_t = kernel_selector::histc_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::histc_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<histc_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<histc>();
        auto params = get_default_params<kernel_selector::histc_params>(impl_param);

        params.bins = primitive->bins;
        params.min_val = primitive->min_val;
        params.max_val = primitive->max_val;

        return params;
    }
};

namespace detail {

attach_histc_impl::attach_histc_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {format::bfyx, format::bfzyx, format::bfwzyx};
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<histc>::add(impl_types::ocl, typed_primitive_impl_ocl<histc>::create<histc_impl>, keys);
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::histc_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::histc)
