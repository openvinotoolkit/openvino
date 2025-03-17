// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "primitive_base.hpp"
#include "feed_forward/feed_forward_kernel_ref.h"
#include "feed_forward/feed_forward_kernel_selector.h"
#include "feed_forward_inst.h"

namespace cldnn {
namespace ocl {

struct feed_forward_impl : typed_primitive_impl_ocl<feed_forward> {
    using parent = typed_primitive_impl_ocl<feed_forward>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::feed_forward_kernel_selector;
    using kernel_params_t = kernel_selector::feed_forward_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::feed_forward_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<feed_forward_impl, kernel_params_t>(*this);
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
        auto params = get_default_params<kernel_selector::feed_forward_params>(impl_param, is_shape_agnostic);

        auto data_inputs_num = impl_param.input_layouts.size();
        params.inputs.resize(data_inputs_num);
        for (size_t i = 0; i < data_inputs_num; i++) {
            params.inputs[i] = convert_data_tensor(impl_param.get_input_layout(i));
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

attach_feed_forward_impl::attach_feed_forward_impl() {
    auto types = {
        data_types::f32,
        data_types::f16
    };

    auto formats = {
        format::bfyx,
        format::bfzyx
    };

    implementation_map<feed_forward>::add(impl_types::ocl,
                                    shape_types::any,
                                    typed_primitive_impl_ocl<feed_forward>::create<feed_forward_impl>,
                                    types,
                                    formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::feed_forward_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::feed_forward)
