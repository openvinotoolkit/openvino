// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "convert_color_inst.h"
#include "convert_color/convert_color_kernel_selector.h"
#include "convert_color/convert_color_kernel_base.h"

namespace cldnn {
namespace ocl {
struct convert_color_impl : typed_primitive_impl_ocl<convert_color> {
    using parent = typed_primitive_impl_ocl<convert_color>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::convert_color_kernel_selector;
    using kernel_params_t = kernel_selector::convert_color_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::convert_color_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<convert_color_impl, kernel_params_t>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<convert_color>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<convert_color>();

        auto params = get_default_params<kernel_selector::convert_color_params>(impl_param);

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.input_color_format = static_cast<kernel_selector::color_format>(primitive->input_color_format);
        params.output_color_format = static_cast<kernel_selector::color_format>(primitive->output_color_format);
        params.mem_type = static_cast<kernel_selector::memory_type>(primitive->mem_type);

        return params;
    }
};

namespace detail {

attach_convert_color_impl::attach_convert_color_impl() {
    implementation_map<convert_color>::add(impl_types::ocl, typed_primitive_impl_ocl<convert_color>::create<convert_color_impl>, {
        std::make_tuple(data_types::f32, format::nv12),
        std::make_tuple(data_types::f16, format::nv12),
        std::make_tuple(data_types::u8,  format::nv12),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8,  format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::convert_color_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::convert_color)
