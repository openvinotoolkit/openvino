// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "primitive_base.hpp"
#include "group_normalization_inst.h"
#include "group_normalization/group_normalization_kernel_ref.h"
#include "group_normalization/group_normalization_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct group_normalization_impl : typed_primitive_impl_ocl<group_normalization> {
    using parent = typed_primitive_impl_ocl<group_normalization>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::group_normalization_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::group_normalization_params, kernel_selector::group_normalization_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::group_normalization_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<group_normalization_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<group_normalization>();
        auto params = get_default_params<kernel_selector::group_normalization_params>(impl_param, is_shape_agnostic);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
        auto optional_params = get_default_optional_params<kernel_selector::group_normalization_optional_params>(impl_param.get_program());
        params.num_groups = primitive->num_groups;
        params.epsilon = primitive->epsilon;
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_group_normalization_impl::attach_group_normalization_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
            format::bfyx,
            format::byxf,
            format::yxfb,
            format::bfzyx,
            format::b_fs_yx_fsv2,
            format::b_fs_zyx_fsv2,
            format::b_fs_yx_fsv4,
            format::b_fs_zyx_fsv4,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
    };

    implementation_map<group_normalization>::add(impl_types::ocl, shape_types::static_shape,
                                     typed_primitive_impl_ocl<group_normalization>::create<group_normalization_impl>,
                                     types,
                                     formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::group_normalization_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
