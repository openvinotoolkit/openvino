// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "random_uniform_inst.h"
#include "random_uniform/random_uniform_kernel_ref.h"
#include "random_uniform/random_uniform_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct random_uniform_impl : typed_primitive_impl_ocl<random_uniform> {
    using parent = typed_primitive_impl_ocl<random_uniform>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::random_uniform_kernel_selector;
    using kernel_params_t = kernel_selector::random_uniform_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::random_uniform_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<random_uniform_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<random_uniform>();
        auto params = get_default_params<kernel_selector::random_uniform_params>(impl_param);
        params.global_seed = primitive->global_seed;
        params.op_seed = primitive->op_seed;
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));

        return params;
    }
};

namespace detail {

attach_random_uniform_impl::attach_random_uniform_impl() {
    implementation_map<random_uniform>::add(impl_types::ocl, typed_primitive_impl_ocl<random_uniform>::create<random_uniform_impl>, {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f16, format::bfzyx),
            std::make_tuple(data_types::f16, format::bfwzyx),
            std::make_tuple(data_types::f32, format::bfyx),
            std::make_tuple(data_types::f32, format::bfzyx),
            std::make_tuple(data_types::f32, format::bfwzyx),
            std::make_tuple(data_types::i32, format::bfyx),
            std::make_tuple(data_types::i32, format::bfzyx),
            std::make_tuple(data_types::i32, format::bfwzyx),
            std::make_tuple(data_types::i64, format::bfyx),
            std::make_tuple(data_types::i64, format::bfzyx),
            std::make_tuple(data_types::i64, format::bfwzyx),
    });
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::random_uniform_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::random_uniform)
