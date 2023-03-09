// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "batch_to_space_inst.h"
#include "batch_to_space/batch_to_space_kernel_selector.h"
#include "batch_to_space/batch_to_space_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct batch_to_space_impl : typed_primitive_impl_ocl<batch_to_space> {
    using parent = typed_primitive_impl_ocl<batch_to_space>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::batch_to_space_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::batch_to_space_params, kernel_selector::batch_to_space_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<batch_to_space_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<batch_to_space>();
        auto params = get_default_params<kernel_selector::batch_to_space_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::batch_to_space_optional_params>(impl_param.get_program());

        params.block_shape = convert_dim_vector(primitive->block_shape);
        params.crops_begin = convert_dim_vector(primitive->crops_begin);
        params.crops_end = convert_dim_vector(primitive->crops_end);

        return {params, optional_params};
    }
};

namespace detail {

attach_batch_to_space_impl::attach_batch_to_space_impl() {
    implementation_map<batch_to_space>::add(impl_types::ocl, typed_primitive_impl_ocl<batch_to_space>::create<batch_to_space_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::batch_to_space_impl)
