// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gather_nd_inst.h"
#include "gather/gather_nd_kernel_selector.h"
#include "gather/gather_nd_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct gather_nd_impl : typed_primitive_impl_ocl<gather_nd> {
    using parent = typed_primitive_impl_ocl<gather_nd>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_nd_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gather_nd_params, kernel_selector::gather_nd_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<gather_nd>();
        auto params = get_default_params<kernel_selector::gather_nd_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::gather_nd_optional_params>(impl_param.get_program());

        params.indices_rank = primitive->indices_rank;
        params.batch_dims = primitive->batch_dims;
        params.batch_merged_output = primitive->batch_merged_output;

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        return {params, optional_params};
    }
};

namespace detail {

attach_gather_nd_impl::attach_gather_nd_impl() {
    implementation_map<gather_nd>::add(impl_types::ocl, typed_primitive_impl_ocl<gather_nd>::create<gather_nd_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nd_impl)
