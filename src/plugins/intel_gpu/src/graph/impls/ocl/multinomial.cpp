// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "primitive_base.hpp"
#include "multinomial_inst.h"
#include "multinomial/multinomial_kernel_ref.h"
#include "multinomial/multinomial_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct multinomial_impl : typed_primitive_impl_ocl<multinomial> {
    using parent = typed_primitive_impl_ocl<multinomial>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::multinomial_kernel_selector;
    using kernel_params_t = kernel_selector::multinomial_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::multinomial_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<multinomial_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<multinomial>();
        auto params = get_default_params<kernel_selector::multinomial_params>(impl_param, is_shape_agnostic);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.output_data_type = primitive->output_data_type;
        params.with_replacement = primitive->with_replacement;
        params.log_probs = primitive->log_probs;
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_multinomial_impl::attach_multinomial_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32};
    implementation_map<multinomial>::add(impl_types::ocl, shape_types::static_shape,
                                     typed_primitive_impl_ocl<multinomial>::create<multinomial_impl>,
                                     types,
                                     {format::bfyx});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::multinomial_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::multinomial)
