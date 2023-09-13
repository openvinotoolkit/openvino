// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "range_inst.h"
#include "range/range_kernel_selector.h"
#include "range/range_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct range_impl : typed_primitive_impl_ocl<range> {
    using parent = typed_primitive_impl_ocl<range>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::range_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::range_params, kernel_selector::range_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::range_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<range_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::range_params>(impl_param, is_shape_agnostic);
        for (int i : {1, 2})
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        auto optional_params = get_default_optional_params<kernel_selector::range_optional_params>(impl_param.get_program());

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
       auto kernel_params = get_kernel_params(impl_param, true);
       (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_range_impl::attach_range_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8
    };

    auto formats = {
        format::bfyx
    };

    implementation_map<range>::add(impl_types::ocl,
                                   shape_types::any,
                                   typed_primitive_impl_ocl<range>::create<range_impl>,
                                   types,
                                   formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::range_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::range)
