// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "normalize_inst.h"
#include "normalize/normalize_kernel_selector.h"
#include "normalize/normalize_kernel_base.h"

namespace cldnn {
namespace ocl {

struct normalize_impl : typed_primitive_impl_ocl<normalize> {
    using parent = typed_primitive_impl_ocl<normalize>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::normalize_kernel_selector;
    using kernel_params_t = kernel_selector::normalize_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::normalize_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<normalize_impl, kernel_params_t>(*this);
    }

protected:
     kernel_arguments_data get_arguments(const typed_primitive_inst<normalize>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        args.scale_table = instance.scale_memory();
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<normalize>();
        auto params = get_default_params<kernel_selector::normalize_params>(impl_param);

        auto scale_layout = impl_param.get_input_layout(1);

        params.normMode = primitive->across_spatial ? kernel_selector::normalize_mode::ACROSS_SPATIAL
                                                    : kernel_selector::normalize_mode::WITHIN_SPATIAL;
        params.epsilon = primitive->epsilon;
        if (format::is_simple_data_format(scale_layout.format)) {
            params.scaleTable = convert_data_tensor(scale_layout).FlattenFeatureAndSpatials();
        } else {
            params.scaleTable = convert_data_tensor(scale_layout);
        }
        return params;
    }
};

namespace detail {

attach_normalize_impl::attach_normalize_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8};
    auto formats = {
        format::bfyx,
        format::yxfb,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
    };
    implementation_map<normalize>::add(impl_types::ocl, typed_primitive_impl_ocl<normalize>::create<normalize_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::normalize_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::normalize)
