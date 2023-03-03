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

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<select_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::select_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::select_optional_params>(impl_param.get_program());

        std::vector<layout> input_layouts = impl_param.input_layouts;
        auto o_layout = impl_param.get_output_layout();

        auto broadcastable = [&](const layout& a, const layout& b) {
            if (a.is_dynamic() || b.is_dynamic()) {
                return false;
            }

            auto dims_a = a.get_partial_shape();
            auto dims_b = b.get_partial_shape();

            size_t min_size = std::min(dims_a.size(), dims_b.size());

            for (size_t i = 0; i < min_size; ++i) {
                if (!(dims_a[i] == 1 || dims_b[i] == 1 || dims_a[i] == dims_b[i])) {
                    return false;
                }
            }
            return true;
        };

        for (auto& l : input_layouts) {
            auto pshape = l.get_partial_shape();
            auto rank = pshape.size();

            if (rank < 4 && !broadcastable(o_layout, l)) {
                pshape.insert(pshape.begin(), 4 - rank, 1);
                layout new_layout = l;
                new_layout.set_partial_shape(pshape);
                l = new_layout;
            }
        }

        for (size_t i = 1; i < input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(input_layouts[i]));
        }
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        update_kernels_list_to_skip();
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
