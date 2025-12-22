// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "random_uniform_inst.h"
#include "random_uniform/random_uniform_kernel_ref.h"
#include "random_uniform/random_uniform_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct random_uniform_impl : typed_primitive_impl_ocl<random_uniform> {
    using parent = typed_primitive_impl_ocl<random_uniform>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::random_uniform_kernel_selector;
    using kernel_params_t = kernel_selector::random_uniform_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::random_uniform_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<random_uniform_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<random_uniform>();
        auto params = get_default_params<kernel_selector::random_uniform_params>(impl_param, is_shape_agnostic);
        params.global_seed = primitive->global_seed;
        params.op_seed = primitive->op_seed;
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

attach_random_uniform_impl::attach_random_uniform_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<random_uniform>::add(impl_types::ocl,
                                            shape_types::static_shape,
                                            typed_primitive_impl_ocl<random_uniform>::create<random_uniform_impl>,
                                            types,
                                            formats);

    implementation_map<random_uniform>::add(impl_types::ocl,
                                            shape_types::dynamic_shape,
                                            typed_primitive_impl_ocl<random_uniform>::create<random_uniform_impl>,
                                            types,
                                            formats);
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::random_uniform_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::random_uniform)
