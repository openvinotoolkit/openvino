// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gather_tree_inst.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"

namespace cldnn {
namespace ocl {

struct gather_tree_impl : typed_primitive_impl_ocl<gather_tree> {
    using parent = typed_primitive_impl_ocl<gather_tree>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_tree_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gather_tree_params, kernel_selector::gather_tree_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_tree_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::gather_tree_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(impl_param.get_program());

        for (size_t i = 1; i < impl_param.input_layouts.size(); i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        return {params, optional_params};
    }
};

namespace detail {
attach_gather_tree_impl::attach_gather_tree_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32
    };

    auto formats = {
        format::yxfb,
        format::bfyx,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    implementation_map<gather_tree>::add(impl_types::ocl, typed_primitive_impl_ocl<gather_tree>::create<gather_tree_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_tree_impl)
