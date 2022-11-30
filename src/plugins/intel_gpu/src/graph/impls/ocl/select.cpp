// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
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

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<select>();
        auto params = get_default_params<kernel_selector::select_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::select_optional_params>(impl_param.get_program());

        std::vector<layout> layouts = impl_param.input_layouts;
        auto o_layout = impl_param.get_output_layout();

        auto broadcastable = [&](layout a, layout b) {
            auto dims_a = a.get_dims();
            auto dims_b = b.get_dims();
            size_t min_size = (dims_a.size() < dims_b.size()) ? dims_a.size(): dims_b.size();

            for (size_t i = 0; i < min_size; i++) {
                if (!(dims_a[i] == 1 || dims_b[i] == 1 || dims_a[i] == dims_b[i])) {
                    return false;
                }
            }
            return true;
        };

        for (size_t i = 0; i < layouts.size(); i++) {
            auto shape = layouts[i].get_shape();
            auto shape_size = shape.size();
            if (shape_size < 4 && !broadcastable(o_layout, layouts[i])) {
                shape.insert(shape.begin(), 4 - shape_size, 1);
                layout new_layout = layouts[i];
                new_layout.set_partial_shape(shape);
                layouts[i] = new_layout;
            }
        }

        for (size_t i = 1; i < layouts.size(); i++) {
            params.inputs.push_back(convert_data_tensor(layouts[i]));
        }
        return {params, optional_params};
    }
};

namespace detail {

attach_select_impl::attach_select_impl() {
    implementation_map<select>::add(impl_types::ocl, typed_primitive_impl_ocl<select>::create<select_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
    });

    impl_hash_key<select>::add(typed_primitive_impl_ocl<select>::get_impl_key<select_impl>);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::select_impl)
