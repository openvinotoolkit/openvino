// Copyright (C) 2018-2025 Intel Corporation
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
    using kernel_params_t = kernel_selector::batch_to_space_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::batch_to_space_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<batch_to_space_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<batch_to_space>();
        auto params = get_default_params<kernel_selector::batch_to_space_params>(impl_param);

        if (primitive->shape_constant) {
            params.block_type = kernel_selector::base_params::ArgType::Constant;
            params.block_shape = convert_dim_vector(primitive->block_shape);

            params.begin_type = kernel_selector::base_params::ArgType::Constant;
            params.crops_begin = convert_dim_vector(primitive->crops_begin);

            params.end_type = kernel_selector::base_params::ArgType::Constant;
            params.crops_end = convert_dim_vector(primitive->crops_end);
        } else {
            params.block_input_index = 1;
            params.block_type = kernel_selector::base_params::ArgType::Input;
            auto block_layout = impl_param.get_input_layout(params.block_input_index);
            params.inputs.push_back(convert_data_tensor(block_layout));
            params.block_dims = block_layout.count();

            params.begin_input_index = 2;
            params.begin_type = kernel_selector::base_params::ArgType::Input;
            auto begin_layout = impl_param.get_input_layout(params.begin_input_index);
            params.inputs.push_back(convert_data_tensor(begin_layout));
            params.begin_dims = begin_layout.count();

            params.end_input_index = 3;
            params.end_type = kernel_selector::base_params::ArgType::Input;
            auto end_layout = impl_param.get_input_layout(params.end_input_index);
            params.inputs.push_back(convert_data_tensor(end_layout));
            params.end_dims = end_layout.count();
        }

        return params;
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
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::batch_to_space)
