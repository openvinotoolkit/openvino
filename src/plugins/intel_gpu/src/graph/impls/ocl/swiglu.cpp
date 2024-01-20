// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "swiglu_inst.h"
#include "swiglu/swiglu_kernel_selector.h"
#include "swiglu/swiglu_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct swiglu_impl : typed_primitive_impl_ocl<swiglu> {
    using parent = typed_primitive_impl_ocl<swiglu>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::swiglu_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::swiglu_params, kernel_selector::swiglu_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::swiglu_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<swiglu_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<swiglu>();
        auto params = get_default_params<kernel_selector::swiglu_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::swiglu_optional_params>(impl_param.get_program());

        params.axis = primitive->axis;
        params.split_length = primitive->split_length;

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_swiglu_impl::attach_swiglu_impl() {
    implementation_map<swiglu>::add(impl_types::ocl,
                                    shape_types::any,
                                    typed_primitive_impl_ocl<swiglu>::create<swiglu_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
