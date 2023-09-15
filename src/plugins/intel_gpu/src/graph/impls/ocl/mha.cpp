// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "mha_inst.h"
#include "mha/mha_kernel_selector.h"
#include "mha/mha_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct mha_impl : typed_primitive_impl_ocl<mha> {
    using parent = typed_primitive_impl_ocl<mha>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::mha_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::mha_params, kernel_selector::mha_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::mha_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mha_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<mha>();
        auto params = get_default_params<kernel_selector::mha_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::mha_optional_params>(impl_param.get_program());

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
        return {params, optional_params};
    }
};

namespace detail {

attach_mha_impl::attach_mha_impl() {
    auto types = {data_types::f16};
    auto formats = {
            format::bfyx,
    };

    implementation_map<mha>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<mha>::create<mha_impl>,
        types,
        formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mha_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mha)
