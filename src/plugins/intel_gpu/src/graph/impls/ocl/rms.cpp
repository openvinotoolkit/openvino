// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "rms_inst.h"
#include "rms/rms_kernel_selector.h"
#include "rms/rms_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct rms_impl : typed_primitive_impl_ocl<rms> {
    using parent = typed_primitive_impl_ocl<rms>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::rms_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::rms_params, kernel_selector::rms_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rms_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rms_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<rms>();
        auto params = get_default_params<kernel_selector::rms_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::rms_optional_params>(impl_param.get_program());

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.epsilon = primitive->epsilon;
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_rms_impl::attach_rms_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32
    };

    auto formats = {
        format::bfyx,
        format::bfzyx
    };

    implementation_map<rms>::add(impl_types::ocl,
                                 shape_types::any,
                                 typed_primitive_impl_ocl<rms>::create<rms_impl>,
                                 types,
                                 formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rms_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rms)
