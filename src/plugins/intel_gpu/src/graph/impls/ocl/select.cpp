// Copyright (C) 2018-2026 Intel Corporation
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
    using kernel_params_t = kernel_selector::select_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::select_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<select_impl, kernel_params_t>(*this);
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
        auto params = get_default_params<kernel_selector::select_params>(impl_param, is_shape_agnostic);

        std::vector<layout> input_layouts = impl_param.input_layouts;

        for (size_t i = 1; i < input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(input_layouts[i]));
        }
        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        auto& output_layout = updated_impl_params.output_layouts[0];
        auto out_pshape = output_layout.get_partial_shape();
        size_t target_rank = std::max<size_t>(4, out_pshape.size());

        for (auto& input_layout : updated_impl_params.input_layouts) {
            auto input_pshape = input_layout.get_partial_shape();
            input_pshape = extend_shape_to_rank_from_begin(input_pshape, target_rank);
            input_layout.set_partial_shape(input_pshape);
            input_layout.format = format::adjust_to_rank(input_layout.format, input_pshape.size());
        }

        output_layout.set_partial_shape(extend_shape_to_rank_from_begin(out_pshape, target_rank));
        output_layout.format = format::adjust_to_rank(output_layout.format, target_rank);

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
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

attach_select_impl::attach_select_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i8,
        data_types::u8
    };

    auto static_formats = {
        format::bfyx,
        format::byxf,
        format::yxfb,
        format::bfzyx,
    };

    implementation_map<select>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<select>::create<select_impl>,
                                    types,
                                    static_formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
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
