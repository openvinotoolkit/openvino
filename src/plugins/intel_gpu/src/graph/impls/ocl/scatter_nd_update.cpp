// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "scatter_nd_update_inst.h"
#include "scatter_update/scatter_nd_update_kernel_selector.h"
#include "scatter_update/scatter_nd_update_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct scatter_nd_update_impl : typed_primitive_impl_ocl<scatter_nd_update> {
    using parent = typed_primitive_impl_ocl<scatter_nd_update>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::scatter_nd_update_kernel_selector;
    using kernel_params_t = kernel_selector::scatter_nd_update_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::scatter_nd_update_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scatter_nd_update_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<scatter_nd_update>();
        auto params = get_default_params<kernel_selector::scatter_nd_update_params>(impl_param, is_shape_agnostic);

        params.indices_rank = primitive->indices_rank;

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));

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

attach_scatter_nd_update_impl::attach_scatter_nd_update_impl() {
    auto types = { data_types::f32, data_types::f16, data_types::i32, data_types::i8, data_types::u8 };
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv4,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,

        format::bfwzyx
    };

    implementation_map<scatter_nd_update>::add(impl_types::ocl,
                                               shape_types::static_shape,
                                               typed_primitive_impl_ocl<scatter_nd_update>::create<scatter_nd_update_impl>,
                                               types,
                                               formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<scatter_nd_update>::add(impl_types::ocl,
                                               shape_types::dynamic_shape,
                                               typed_primitive_impl_ocl<scatter_nd_update>::create<scatter_nd_update_impl>,
                                               types,
                                               dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scatter_nd_update_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scatter_nd_update)
