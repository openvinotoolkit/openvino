// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "unique/unique_kernel_ref.hpp"
#include "unique/unique_kernel_selector.hpp"
#include "unique_inst.hpp"

namespace cldnn {
namespace ocl {

struct unique_impl : typed_primitive_impl_ocl<unique> {
    using parent = typed_primitive_impl_ocl<unique>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::unique_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::unique_params, kernel_selector::unique_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<unique_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<unique>();
        auto params = get_default_params<kernel_selector::unique_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::unique_optional_params>(impl_param.get_program());

        params.flattened = primitive->flattened;
        params.axis = primitive->axis;
        params.sorted = primitive->sorted;

        for (auto i = 1U; i < impl_param.output_layouts.size(); ++i) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts.at(i)));
        }

        return {params, optional_params};
    }
};

namespace detail {

attach_unique_impl::attach_unique_impl() {
    auto types = {
        data_types::u8,
        data_types::i8,
        data_types::f16,
        data_types::f32,
        data_types::i32,
        data_types::i64,
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<unique>::add(impl_types::ocl,
                                    typed_primitive_impl_ocl<unique>::create<unique_impl>,
                                    types,
                                    formats);
}
}  // namespace detail

struct unique_reshape_impl : typed_primitive_impl_ocl<unique_reshape> {
    using parent = typed_primitive_impl_ocl<unique_reshape>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::unique_reshape_kernel_selector;
    using kernel_params_t =
        std::pair<kernel_selector::unique_reshape_params, kernel_selector::unique_reshape_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<unique_reshape_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<unique_reshape>();
        auto params = get_default_params<kernel_selector::unique_reshape_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::unique_reshape_optional_params>(impl_param.get_program());

        params.flattened = primitive->flattened;
        params.axis = primitive->axis;

        for (auto i = 1U; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts.at(i)));
        }

        for (auto i = 1U; i < impl_param.output_layouts.size(); ++i) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts.at(i)));
        }

        return {params, optional_params};
    }
};

namespace detail {

attach_unique_reshape_impl::attach_unique_reshape_impl() {
    auto types = {
        data_types::u8,
        data_types::i8,
        data_types::f16,
        data_types::f32,
        data_types::i32,
        data_types::i64,
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<unique_reshape>::add(impl_types::ocl,
                                            typed_primitive_impl_ocl<unique_reshape>::create<unique_reshape_impl>,
                                            types,
                                            formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::unique_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::unique_reshape_impl)
