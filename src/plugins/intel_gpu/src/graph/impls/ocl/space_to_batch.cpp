// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "space_to_batch_inst.h"
#include "space_to_batch/space_to_batch_kernel_selector.h"
#include "space_to_batch/space_to_batch_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct space_to_batch_impl : typed_primitive_impl_ocl<space_to_batch> {
    using parent = typed_primitive_impl_ocl<space_to_batch>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::space_to_batch_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::space_to_batch_params, kernel_selector::space_to_batch_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<space_to_batch_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<space_to_batch>();
        auto params = get_default_params<kernel_selector::space_to_batch_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::space_to_batch_optional_params>(impl_param.get_program());

        params.block_shape = convert_dim_vector(primitive->block_shape);
        params.pads_begin = convert_dim_vector(primitive->pads_begin);
        params.pads_end = convert_dim_vector(primitive->pads_end);

        return {params, optional_params};
    }
};

namespace detail {

attach_space_to_batch_impl::attach_space_to_batch_impl() {
    implementation_map<space_to_batch>::add(impl_types::ocl, typed_primitive_impl_ocl<space_to_batch>::create<space_to_batch_impl>, {
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
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::space_to_batch_impl)
