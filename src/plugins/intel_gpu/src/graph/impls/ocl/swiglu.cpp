// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "primitive_base.hpp"
#include "swiglu/swiglu_kernel_ref.h"
#include "swiglu/swiglu_kernel_selector.h"
#include "swiglu_inst.h"

namespace cldnn {
namespace ocl {

struct swiglu_impl : typed_primitive_impl_ocl<swiglu> {
    using parent = typed_primitive_impl_ocl<swiglu>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::swiglu_kernel_selector;
    using kernel_params_t = kernel_selector::swiglu_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::swiglu_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<swiglu_impl>(*this);
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
        const auto& primitive = impl_param.typed_desc<swiglu>();
        auto params = get_default_params<kernel_selector::swiglu_params>(impl_param, is_shape_agnostic);

        auto rank = impl_param.get_input_layout(0).get_partial_shape().rank();
        params.axis = ov::util::normalize(primitive->axis, rank.get_length());
        params.split_length = primitive->split_lengths;

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_swiglu_impl::attach_swiglu_impl() {
    auto types = {
        data_types::f32,
        data_types::f16
    };

    auto formats = {
        format::bfyx,
        format::bfzyx
    };

    implementation_map<swiglu>::add(impl_types::ocl,
                                    shape_types::any,
                                    typed_primitive_impl_ocl<swiglu>::create<swiglu_impl>,
                                    types,
                                    formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::swiglu_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::swiglu)
