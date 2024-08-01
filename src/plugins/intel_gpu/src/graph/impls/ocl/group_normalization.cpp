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
    using kernel_params_t = kernel_selector::group_normalization_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::group_normalization_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<group_normalization_impl>(*this);
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
        const auto& primitive = impl_param.typed_desc<group_normalization>();
        auto params = get_default_params<kernel_selector::group_normalization_params>(impl_param, is_shape_agnostic);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
        params.num_groups = primitive->num_groups;
        params.epsilon = primitive->epsilon;
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

attach_group_normalization_impl::attach_group_normalization_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
            format::bfyx,
            format::bfzyx,
            format::b_fs_yx_fsv16,
    };

    implementation_map<group_normalization>::add(impl_types::ocl, shape_types::static_shape,
                                     typed_primitive_impl_ocl<group_normalization>::create<group_normalization_impl>,
                                     types,
                                     formats);

    const std::vector<format::type> dyn_formats {
        format::bfyx,
        format::b_fs_yx_fsv16,
    };

    implementation_map<group_normalization>::add(impl_types::ocl, shape_types::dynamic_shape,
                                                 typed_primitive_impl_ocl<group_normalization>::create<group_normalization_impl>,
                                                 types, dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::group_normalization_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::group_normalization)
