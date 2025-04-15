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
        return make_deep_copy<swiglu_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
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
        params.glu_type = primitive->glu_type;
        params.split_to_glu_idx = static_cast<int32_t>(primitive->split_to_glu_idx);

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
