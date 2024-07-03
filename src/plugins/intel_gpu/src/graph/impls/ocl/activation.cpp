// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "activation_inst.h"
#include "activation/activation_kernel_base.h"
#include "activation/activation_kernel_selector.h"

namespace {
inline void convert_new_activation_func(const cldnn::activation& prim, std::vector<kernel_selector::base_activation_params>& params) {
    params.insert(params.begin(), {get_kernel_selector_activation_param(prim.activation_function),
                                   prim.additional_params.a,
                                   prim.additional_params.b});
}
}  // namespace

namespace cldnn {
namespace ocl {

struct activation_impl : typed_primitive_impl_ocl<activation> {
    using parent = typed_primitive_impl_ocl<activation>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::activation_kernel_selector;
    using kernel_params_t = kernel_selector::activation_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::activation_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<activation_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<activation>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        if (instance.is_parameterized()) {
            args.slope = instance.slope_memory();
        }

        return args;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<activation>();
        auto params = get_default_params<kernel_selector::activation_params>(impl_param, is_shape_agnostic);

        convert_new_activation_func(*primitive, params.activations);

        bool is_parameterized = !primitive->additional_params_input.empty();
        if (is_parameterized) {
            const auto& slope_layout = impl_param.input_layouts[1];
            const auto& output_layout = impl_param.get_output_layout();

            if (!impl_param.is_dynamic()) {
                const auto params_num = kernel_selector::GetActivationAdditionalParamsNumber(params.activations[0].function);
                OPENVINO_ASSERT(slope_layout.count() >= static_cast<size_t>(output_layout.feature() * params_num),
                                "[GPU] Invalid slope size in ", primitive->id);
            }
            params.inputActivationParams.push_back(convert_data_tensor(slope_layout));
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

attach_activation_impl::attach_activation_impl() {
     auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    auto static_formats = {
        format::yxfb,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    auto keys = implementation_map<activation>::combine(types, static_formats);
    keys.emplace(data_types::f16, format::fs_b_yx_fsv32);

    implementation_map<activation>::add(impl_types::ocl,
                                        shape_types::dynamic_shape,
                                        typed_primitive_impl_ocl<activation>::create<activation_impl>,
                                        types,
                                        dyn_formats);

    implementation_map<activation>::add(impl_types::ocl,
                                        shape_types::static_shape,
                                        typed_primitive_impl_ocl<activation>::create<activation_impl>,
                                        keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::activation_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::activation)
