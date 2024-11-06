// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "search_sorted_inst.h"
#include "search_sorted/search_sorted_kernel_selector.h"
#include "search_sorted/search_sorted_kernel_base.h"

namespace cldnn {
namespace ocl {

struct search_sorted_impl : typed_primitive_impl_ocl<search_sorted> {
    using parent = typed_primitive_impl_ocl<search_sorted>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::search_sorted_kernel_selector;
    using kernel_params_t = kernel_selector::search_sorted_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::search_sorted_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<search_sorted_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<search_sorted>();
        auto params = get_default_params<kernel_selector::search_sorted_params>(impl_param);

        params.right_mode = primitive->right_mode;
        return params;
    }
};

namespace detail {

attach_search_sorted_impl::attach_search_sorted_impl() {
    implementation_map<search_sorted>::add(impl_types::ocl, typed_primitive_impl_ocl<search_sorted>::create<search_sorted_impl>, {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::search_sorted_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::search_sorted)
