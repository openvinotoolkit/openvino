// Copyright (C) 2018-2025 Intel Corporation
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
    using kernel_params_t = kernel_selector::space_to_batch_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::space_to_batch_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<space_to_batch_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<space_to_batch>();
        auto params = get_default_params<kernel_selector::space_to_batch_params>(impl_param);

        if (primitive->shape_constant) {
            params.block_type = kernel_selector::base_params::ArgType::Constant;
            params.block_shape = convert_dim_vector(primitive->block_shape);

            params.begin_type = kernel_selector::base_params::ArgType::Constant;
            params.pads_begin = convert_dim_vector(primitive->pads_begin);

            params.end_type = kernel_selector::base_params::ArgType::Constant;
            params.pads_end = convert_dim_vector(primitive->pads_end);
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
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::space_to_batch)
