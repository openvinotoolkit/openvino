// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "rope_inst.h"
#include "rope/rope_kernel_selector.h"
#include "rope/rope_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct rope_impl : typed_primitive_impl_ocl<rope> {
    using parent = typed_primitive_impl_ocl<rope>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::rope_kernel_selector;
    using kernel_params_t = kernel_selector::rope_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rope_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rope_impl>(*this);
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
        const auto& primitive = impl_param.typed_desc<rope>();
        auto params = get_default_params<kernel_selector::rope_params>(impl_param, is_shape_agnostic);

        params.head_cnt = primitive->config.head_cnt;
        params.head_size = primitive->config.head_size;
        params.rotary_ndims = primitive->config.rotary_ndims;

        params.slice_start = primitive->config.slice_start;
        params.slice_stop = primitive->config.slice_stop;

        params.axis = primitive->config.is_qwen || primitive->config.is_chatglm ? 2 : 3;
        params.num_of_inputs = primitive->config.is_chatglm || primitive->config.is_interleaved ? 2 : 3;

        params.is_qwen = primitive->config.is_qwen;
        params.is_chatglm = primitive->config.is_chatglm;

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_rope_impl::attach_rope_impl() {
    auto types = {
        data_types::f32,
        data_types::f16
    };

    auto formats = {
        format::bfyx
    };

    implementation_map<rope>::add(impl_types::ocl,
                                  shape_types::any,
                                  typed_primitive_impl_ocl<rope>::create<rope_impl>,
                                  types,
                                  formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rope_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rope)
