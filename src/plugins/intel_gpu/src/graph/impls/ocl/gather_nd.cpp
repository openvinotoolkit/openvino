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

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_nd_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
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

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_gather_nd_impl::attach_gather_nd_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32
    };

    auto static_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<gather_nd>::add(impl_types::ocl,
                                       shape_types::static_shape,
                                       typed_primitive_impl_ocl<gather_nd>::create<gather_nd_impl>,
                                       types,
                                       static_formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<gather_nd>::add(impl_types::ocl,
                                       shape_types::dynamic_shape,
                                       typed_primitive_impl_ocl<gather_nd>::create<gather_nd_impl>,
                                       types,
                                       dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nd_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_nd)
