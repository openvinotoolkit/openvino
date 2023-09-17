// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "select_inst.h"
#include "select/select_kernel_selector.h"
#include "select/select_kernel_base.h"

namespace cldnn {
namespace ocl {

struct select_impl : typed_primitive_impl_ocl<select> {
    using parent = typed_primitive_impl_ocl<select>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::select_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::select_params, kernel_selector::select_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::select_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<select_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::select_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::select_optional_params>(impl_param.get_program());

        std::vector<layout> input_layouts = impl_param.input_layouts;

        for (size_t i = 1; i < input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(input_layouts[i]));
        }
        return {params, optional_params};
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
        }

        auto& output_layout = updated_impl_params.output_layouts[0];
        output_layout.set_partial_shape(extend_shape_to_rank_from_begin(output_layout.get_partial_shape()));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_select_impl::attach_select_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8
    };

    auto static_formats = {
        format::bfyx,
        format::byxf,
        format::yxfb,
    };

    implementation_map<select>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<select>::create<select_impl>,
                                    types,
                                    static_formats);

    auto dyn_formats = {
        format::bfyx
    };

    implementation_map<select>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<select>::create<select_impl>,
                                     types,
                                     dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::select_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::select)
