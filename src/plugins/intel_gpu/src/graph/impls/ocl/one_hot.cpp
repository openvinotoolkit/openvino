// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "one_hot_inst.h"
#include "one_hot/one_hot_kernel_selector.h"
#include "one_hot/one_hot_kernel_base.h"

namespace cldnn {
namespace ocl {

struct one_hot_impl : typed_primitive_impl_ocl<one_hot> {
    using parent = typed_primitive_impl_ocl<one_hot>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::one_hot_kernel_selector;
    using kernel_params_t = kernel_selector::one_hot_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::one_hot_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<one_hot_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<one_hot>();
        auto params = get_default_params<kernel_selector::one_hot_params>(impl_param);

        params.one_hot_axis = primitive->one_hot_axis;
        params.on_value = primitive->on_value;
        params.off_value = primitive->off_value;

        auto output_sizes = impl_param.get_output_layout().get_dims();

        params.one_hot_limit = output_sizes[params.one_hot_axis];
        return params;
    }
};

namespace detail {

attach_one_hot_impl::attach_one_hot_impl() {
    implementation_map<one_hot>::add(impl_types::ocl, typed_primitive_impl_ocl<one_hot>::create<one_hot_impl>, {
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::one_hot_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::one_hot)
