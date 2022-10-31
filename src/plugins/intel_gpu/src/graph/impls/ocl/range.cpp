// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <range_inst.h>
#include "primitive_base.hpp"
#include <impls/implementation_map.hpp>
#include <kernel_selector_helper.h>
#include <range/range_kernel_selector.h>
#include <range/range_kernel_ref.h>
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {
namespace ocl {

struct range_impl : typed_primitive_impl_ocl<range> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<range_impl>(*this);
    }

    static std::unique_ptr<primitive_impl> create(const range_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::range_params>(impl_param);
        for (int i : {1, 2})
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        auto optional_params =
            get_default_optional_params<kernel_selector::range_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::range_instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, optional_params);

        return make_unique<range_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_range_impl::attach_range_impl() {
    implementation_map<range>::add(
        impl_types::ocl,
        range_impl::create,
        {
            std::make_tuple(data_types::u8, format::bfyx),
            std::make_tuple(data_types::i8, format::bfyx),
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f32, format::bfyx),
            std::make_tuple(data_types::i32, format::bfyx),
            std::make_tuple(data_types::i64, format::bfyx),
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::range_impl, cldnn::object_type::RANGE_IMPL)
